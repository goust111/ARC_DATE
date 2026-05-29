# Combo variant of 36a08778 (index 25)
# combo_spec: D1/0+D5/2+D14/3
# Name: 对角线下落+固定扩散+入水方向溢出
#
# 设计分析：
# a) 具体化与行为追踪
#    原题是“竖直下落 -> 命中整段架子 -> 沿整段水平铺开 -> 两端都继续竖直下落”。
#    融合后改成：
#    - 水源是 2 像素对角源，例如 (0,4),(1,5)，因此方向可见地指向右下。
#    - 水流每步走 (r+1,c+1) 或 (r+1,c-1)。
#    - 若下一步对角位置正好是架子，则在架子上方一行、以“入水投影列”为中心固定扩散 K 格；
#      也就是说无论架子长短，都会画出同样宽度的水平水带。
#    - 真正继续下落的不是“架子两端”，而是这条固定宽度水带的左端 / 右端 / 两端；
#      选择规则由入水点落在架子左半、右半还是正中决定。
#    小例子：源头朝右下，下一步将撞到第 5 行某架子第 8 列。
#    原题会在第 4 行把整段架子宽度都铺满并从两侧继续竖直落；
#    本变体只会在第 4 行画 [5..11] 这 7 格固定宽度带，若 8 列落在架子左半，
#    则只有第 5 列端点继续向左下走，右端虽然被画出来但不会产生新支流。
#
# b) 复杂度校准
#    只引入三个少量离散自由度：方向只有左右两种、扩散长度固定为单一常数 K、
#    溢出类型只有 left/right/center 三档。这样视觉差异明显，但不会因为连续参数太多而难归纳。
#
# c) 隐式假设审查
#    原 solver 的关键前提都失效了：
#    - “沿同一列向下扫描”失效，必须显式维护 (row,col,dc) 状态。
#    - “命中位置一定在架子正上方同列”失效，命中点改为下一步对角格。
#    - “扩散长度由架子端点决定”失效，改成固定宽度。
#    - “总是双侧继续”失效，必须读取入水点相对架子中线的位置。
#    新 solver 因此只依赖以下新前提：架子是水平连续段；用于固定扩散的那一行在生成器里被预留为空白。
#
# d) 放置合法性
#    生成器先规划主干：首层强制 center 命中形成双分支，再分别给左右子流放 left/right 命中架子。
#    每次放置都会检查：
#    - 对角下落路径不与已有架子、既有水路、源头重叠
#    - 固定宽度扩散带完整位于网格内
#    - 架子段本身不越界、不重叠、不占用预留水路
#
# e) 视觉可观察性
#    每个样例都至少同时出现：
#    - 明显的 45 度斜线
#    - 明显的固定宽度水平带
#    - center 双溢出 + left 单溢出 + right 单溢出
#    因此人类可以从少量样例中看出“宽度固定”和“哪边继续取决于入水位置”。
#
# f) 可视化整洁度
#    生成器拒绝让不同水路复用同一片背景区，避免多层覆盖导致看不清分支来源。
#    也不额外加入干扰架子，优先保持瀑布树结构清楚。
#
# g) 颜色变化可推断性
#    颜色系统不变：背景一种、水一种、架子一种。没有额外颜色映射负担。

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 7
WATER = 6
SHELF = 2
SPREAD = 3


class Source:
    __slots__ = ("row", "col", "dc", "cells")

    def __init__(self, row: int, col: int, dc: int, cells: frozenset[tuple[int, int]]):
        self.row = int(row)
        self.col = int(col)
        self.dc = int(dc)
        self.cells = cells


class Branch:
    __slots__ = ("row", "col", "dc")

    def __init__(self, row: int, col: int, dc: int):
        self.row = int(row)
        self.col = int(col)
        self.dc = int(dc)


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = int(rr + dr), int(cc + dc)
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                            continue
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _detect_roles(grid: np.ndarray) -> tuple[int, int, int]:
    cnt = Counter(map(int, grid.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non_bg = [int(c) for c in cnt if int(c) != int(bg)]
    assert non_bg
    water = int(min(non_bg, key=lambda c: (int(cnt[int(c)]), int(c))))
    shelves = [int(c) for c in non_bg if int(c) != int(water)]
    assert shelves
    return int(bg), int(water), int(shelves[0])


def _extract_sources(grid: np.ndarray, water: int) -> list[Source]:
    sources: list[Source] = []
    for comp in _components8(grid == int(water)):
        if not any(r == 0 for r, _c in comp):
            continue
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        rmax = max(rows)
        bottom_cols = sorted({c for r, c in comp if r == rmax})
        col = int(bottom_cols[len(bottom_cols) // 2])
        top_cols = [c for r, c in comp if r == min(rows)]
        top_center = sum(top_cols) / len(top_cols)
        dc = int(np.sign(col - top_center))
        if dc == 0:
            dc = 1 if col < grid.shape[1] // 2 else -1
        sources.append(
            Source(
                row=int(rmax),
                col=int(col),
                dc=int(dc),
                cells=frozenset((int(r), int(c)) for r, c in comp),
            )
        )
    return sources


def _shelf_segments(grid: np.ndarray, shelf: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    rows: list[list[tuple[int, int]]] = [[] for _ in range(h)]
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != int(shelf):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < w and int(grid[r, c + 1]) == int(shelf):
                c += 1
            rows[r].append((int(c0), int(c)))
            c += 1
    return rows


def _find_segment(segments: list[tuple[int, int]], col: int) -> tuple[int, int] | None:
    for a, b in segments:
        if int(a) <= int(col) <= int(b):
            return (int(a), int(b))
    return None


def _entry_mode(col: int, a: int, b: int) -> str:
    length = int(b - a + 1)
    offset = int(col - a)
    if length % 2 == 1 and offset == length // 2:
        return "center"
    if offset < length / 2:
        return "left"
    return "right"


def _paint_cell(out: np.ndarray, world: np.ndarray, r: int, c: int, water: int, bg: int) -> None:
    if not (0 <= int(r) < out.shape[0] and 0 <= int(c) < out.shape[1]):
        return
    if int(world[r, c]) == int(bg) or int(world[r, c]) == int(water):
        out[int(r), int(c)] = int(water)


def _simulate(grid: np.ndarray, *, collect_events: bool = False):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, water, shelf = _detect_roles(g)
    seg_rows = _shelf_segments(g, shelf)
    sources = _extract_sources(g, water)
    out = g.copy()
    seen: set[tuple[int, int, int]] = set()
    q = deque(Branch(src.row, src.col, src.dc) for src in sources)
    events: list[dict[str, int | str]] = []

    while q:
        state = q.popleft()
        key = (int(state.row), int(state.col), int(state.dc))
        if key in seen:
            continue
        seen.add(key)

        rr, cc, dc = int(state.row), int(state.col), int(state.dc)
        _paint_cell(out, g, rr, cc, water, bg)

        while True:
            nr = int(rr + 1)
            nc = int(cc + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if int(g[nr, nc]) == int(shelf):
                seg = _find_segment(seg_rows[nr], nc)
                if seg is None:
                    break
                a, b = seg
                left = int(nc - SPREAD)
                right = int(nc + SPREAD)
                if not (0 <= left and right < w):
                    break
                clear = True
                for x in range(left, right + 1):
                    if int(g[rr, x]) not in (int(bg), int(water)):
                        clear = False
                        break
                if not clear:
                    break
                for x in range(left, right + 1):
                    out[int(rr), int(x)] = int(water)

                mode = _entry_mode(nc, a, b)
                if collect_events:
                    events.append(
                        {
                            "impact_col": int(nc),
                            "shelf_row": int(nr),
                            "spread_row": int(rr),
                            "left": int(left),
                            "right": int(right),
                            "spread_width": int(right - left + 1),
                            "mode": mode,
                        }
                    )
                if mode in ("left", "center"):
                    q.append(Branch(int(rr), int(left), -1))
                if mode in ("right", "center"):
                    q.append(Branch(int(rr), int(right), 1))
                break

            rr, cc = nr, nc
            _paint_cell(out, g, rr, cc, water, bg)

    if collect_events:
        return out, events
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, collect_events=False)


def _sample_segment(width: int, impact_col: int, mode: str) -> tuple[int, int] | None:
    if mode == "center":
        lengths = [length for length in (5, 7, 9) if length <= width]
        random.shuffle(lengths)
        for length in lengths:
            a = int(impact_col - length // 2)
            b = int(a + length - 1)
            if 0 <= a and b < width and _entry_mode(impact_col, a, b) == "center":
                return (int(a), int(b))
        return None

    for _ in range(80):
        length = random.randint(6, min(10, width))
        a_min = max(0, int(impact_col - length + 1))
        a_max = min(int(impact_col), int(width - length))
        if a_min > a_max:
            continue
        a = random.randint(a_min, a_max)
        b = int(a + length - 1)
        if _entry_mode(impact_col, a, b) == mode:
            return (int(a), int(b))
    return None


def _path_cells(branch: Branch, shelf_row: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(int(branch.row) + 1, int(shelf_row)):
        offset = int(r - branch.row)
        c = int(branch.col + branch.dc * offset)
        cells.append((int(r), int(c)))
    return cells


def _commit_shelf(
    grid: np.ndarray,
    branch: Branch,
    mode: str,
    reserved_bg: set[tuple[int, int]],
    shelf_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[Branch] | None:
    h, w = grid.shape

    for _ in range(120):
        drop = random.randint(2, 4)
        shelf_row = int(branch.row + drop)
        if shelf_row >= h - 1:
            continue

        impact_col = int(branch.col + branch.dc * drop)
        left = int(impact_col - SPREAD)
        right = int(impact_col + SPREAD)
        if not (0 <= left and right < w):
            continue

        seg = _sample_segment(w, impact_col, mode)
        if seg is None:
            continue
        a, b = seg
        spread_row = int(shelf_row - 1)
        route = _path_cells(branch, shelf_row)
        band = [(int(spread_row), int(c)) for c in range(left, right + 1)]
        shelf = [(int(shelf_row), int(c)) for c in range(a, b + 1)]

        ok = True
        for cell in route:
            r, c = cell
            if not (0 <= r < h and 0 <= c < w):
                ok = False
                break
            if cell in blocked or cell in reserved_bg or cell in shelf_cells:
                ok = False
                break
        if not ok:
            continue

        for cell in band:
            if cell in blocked or cell in reserved_bg or cell in shelf_cells:
                ok = False
                break
        if not ok:
            continue

        for cell in shelf:
            if cell in blocked or cell in reserved_bg or cell in shelf_cells:
                ok = False
                break
        if not ok:
            continue

        for cell in route:
            reserved_bg.add(cell)
        for cell in band:
            reserved_bg.add(cell)
        for cell in shelf:
            shelf_cells.add(cell)
            grid[cell] = int(SHELF)

        children: list[Branch] = []
        if mode in ("left", "center"):
            children.append(Branch(int(spread_row), int(left), -1))
        if mode in ("right", "center"):
            children.append(Branch(int(spread_row), int(right), 1))
        return children

    return None


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    out, events = _simulate(grid, collect_events=True)
    modes = Counter(str(event["mode"]) for event in events)
    widths = [int(event["spread_width"]) for event in events]
    return {
        "changed": not np.array_equal(out, grid),
        "event_count": len(events),
        "modes": dict(modes),
        "widths": widths,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 22)
    w = random.randint(12, 22)
    grid = np.full((h, w), int(BG), dtype=int)

    src_dc = random.choice([-1, 1])
    src_col = random.randint(max(4, w // 2 - 2), min(w - 5, w // 2 + 2))
    top_col = int(src_col - src_dc)
    if not (0 <= top_col < w):
        return None
    grid[0, top_col] = int(WATER)
    grid[1, src_col] = int(WATER)

    blocked = {(0, int(top_col)), (1, int(src_col))}
    reserved_bg: set[tuple[int, int]] = set()
    shelf_cells: set[tuple[int, int]] = set()

    root = Branch(1, int(src_col), int(src_dc))
    first = _commit_shelf(grid, root, "center", reserved_bg, shelf_cells, blocked)
    if first is None or len(first) != 2:
        return None

    left_branch = min(first, key=lambda branch: branch.col)
    right_branch = max(first, key=lambda branch: branch.col)
    second_left = _commit_shelf(grid, left_branch, "left", reserved_bg, shelf_cells, blocked)
    if second_left is None or len(second_left) != 1:
        return None
    second_right = _commit_shelf(grid, right_branch, "right", reserved_bg, shelf_cells, blocked)
    if second_right is None or len(second_right) != 1:
        return None

    frontier = second_left + second_right
    random.shuffle(frontier)
    extra_target = random.choice([0, 1, 2])
    for _ in range(extra_target):
        if not frontier:
            break
        branch = frontier.pop(0)
        mode = random.choice(["left", "right", "center"])
        children = _commit_shelf(grid, branch, mode, reserved_bg, shelf_cells, blocked)
        if children is None:
            continue
        frontier.extend(children)

    metrics = _behavior_metrics(grid)
    if not bool(metrics["changed"]):
        return None
    if int(metrics["event_count"]) < 3:
        return None
    widths = list(metrics["widths"])
    if not widths or any(width != 2 * SPREAD + 1 for width in widths):
        return None
    modes = dict(metrics["modes"])
    if modes.get("center", 0) < 1 or modes.get("left", 0) < 1 or modes.get("right", 0) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input for D1/0+D5/2+D14/3 after 400 attempts")
