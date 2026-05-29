# Combo variant of 36a08778 (index 26)
# combo_spec: D10/0+D14/2
# Name: 水源排序处理+交替左右溢出
#
# 设计分析
# - 具体化与行为追踪：
#   原题在同一层架子上总是左右双侧溢出，所以单个水源会长成分叉树；多个水源用同一个 BFS
#   队列并行推进，彼此几乎不相互影响。这里改成两点：
#   1. 多水源按列坐标从小到大依次处理，先处理的水迹会留在输出里，后处理者把这些水迹当障碍。
#   2. 单条水流每撞一次架子只选一个方向，命中第 1/2/3/... 个架子时按左/右/左/... 交替。
#   一个小例子：左源先在第 1 层向左溢出，第 2 层向右溢出，于是在第 2 层 spread row 留下一段
#   水平水迹；右源之后落到同一层时，本来第 1 次也应向左，但左侧路径被这段旧水迹占住，于是
#   改从右侧溢出。结果从原题的“双叉树”变成“先左源 zigzag 占位，后右源被迫绕行”的不对称图形。
# - 复杂度校准：
#   只保留两个离散控制量：水源处理顺序（按离左边缘距离排序）和每流的左右二周期。这样视觉差异
#   很明显，但推理时不需要记更多状态，也不会因为过多自由度导致样例间模式不稳定。
# - 隐式假设审查：
#   原 solver 里 `q = deque(sources)` 默认并行、`seen` 默认允许不同水源共享轨迹、水平扩散默认
#   两边都尝试且只看原图 `g` 是否为背景。融合后这些前提都失效，所以这里改成：
#   1. 外层按排序后的 source 顺序串行处理；
#   2. 垂直和水平推进都看当前 `out`，因此旧水迹会阻挡后续；
#   3. 每次撞架子只尝试一个首选侧，若首选侧被挡再尝试另一侧，防止“只因首选被挡就完全停住”
#      而看不出“绕行”。
# - 放置合法性：
#   生成器显式构造 2 个源和 4 个架子段；共享层上的两个架子段保持至少 1 格背景间隔，避免连成
#   同一段；所有目标溢出点都保证在网格内，且不会把 overflow 点直接放在架子内部。
# - 视觉可观察性：
#   样例里固定生成 2 个水源，并强制出现“左源两次命中形成 zigzag”与“右源第一次本应向左却被
#   旧轨迹挡住、转向右”的现象。只看输入输出即可归纳“先左后右”和“左右交替但可被旧水迹逼改道”。
# - 可视化整洁度：
#   只使用背景/水/架子三色；水迹主要分布在几条清晰的竖线与少量水平段上，共享层只出现一次显式
#   交互，避免网格被过多重叠分支污染。
# - 颜色变化可推断性：
#   本融合不引入新颜色，也不依赖颜色值编码隐藏状态；所有新规则都通过几何结构与先后遮挡体现。

from __future__ import annotations

from collections import Counter, deque
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


def _detect_roles(g: np.ndarray) -> tuple[int, int, int]:
    cnt = Counter(map(int, g.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non = [int(c) for c in cnt if int(c) != int(bg)]
    assert non
    water = int(min(non, key=lambda c: (int(cnt[int(c)]), int(c))))
    other = [int(c) for c in non if int(c) != int(water)]
    assert other
    shelf = int(other[0])
    return bg, water, shelf


def _get_sources(g: np.ndarray, water: int) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for coords in components_mask(g == int(water), connectivity=8):
        if coords.size == 0 or not bool((coords[:, 0] == 0).any()):
            continue
        rmax = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == int(rmax)})
        for c in cols:
            sources.append((int(rmax), int(c)))
    return sorted(sources, key=lambda rc: (int(rc[1]), int(rc[0])))


def _build_shelf_segments(g: np.ndarray, shelf: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    shelf_segs: list[list[tuple[int, int]]] = [[] for _ in range(h)]
    for r in range(h):
        c = 0
        while c < w:
            if int(g[r, c]) != int(shelf):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < w and int(g[r, c + 1]) == int(shelf):
                c += 1
            shelf_segs[r].append((int(c0), int(c)))
            c += 1
    return shelf_segs


def _find_shelf_seg(row_segs: list[tuple[int, int]], c: int) -> tuple[int, int] | None:
    for a, b in row_segs:
        if int(a) <= int(c) <= int(b):
            return (int(a), int(b))
    return None


def _paint_vertical(out: np.ndarray, bg: int, water: int, r: int, c: int) -> None:
    if int(out[r, c]) == int(bg):
        out[r, c] = int(water)


def _try_spill(
    out: np.ndarray,
    bg: int,
    water: int,
    row: int,
    start_c: int,
    target_c: int | None,
) -> tuple[bool, int | None]:
    if target_c is None or int(target_c) == int(start_c):
        return False, None
    step = 1 if int(target_c) > int(start_c) else -1
    for x in range(int(start_c) + step, int(target_c) + step, step):
        if int(out[int(row), int(x)]) != int(bg):
            return False, int(x)
    for x in range(int(start_c) + step, int(target_c) + step, step):
        out[int(row), int(x)] = int(water)
    return True, None


def _run_combo(
    grid: np.ndarray,
    collect_stats: bool = False,
) -> tuple[np.ndarray, dict[str, object] | None]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, water, shelf = _detect_roles(g)
    out = g.copy()
    sources = _get_sources(g, water)
    shelf_segs = _build_shelf_segments(g, shelf)

    stats: dict[str, object] | None = None
    if collect_stats:
        stats = {
            "source_order": [int(c) for _, c in sources],
            "sources": [],
            "rerouted_hits": 0,
            "total_hits": 0,
        }

    for source_idx, (start_r, start_c) in enumerate(sources):
        rr, cc = int(start_r), int(start_c)
        hits = 0
        source_events: list[dict[str, object]] = []
        _paint_vertical(out, bg, water, rr, cc)

        while True:
            nr = int(rr + 1)
            if nr >= h:
                break

            below = int(g[nr, cc])
            if below == int(shelf):
                seg = _find_shelf_seg(shelf_segs[nr], cc)
                if seg is None:
                    break
                a, b = seg
                targets = {
                    "left": int(a - 1) if int(a) > 0 else None,
                    "right": int(b + 1) if int(b) < int(w - 1) else None,
                }
                preferred = "left" if hits % 2 == 0 else "right"
                other = "right" if preferred == "left" else "left"

                chosen_side: str | None = None
                chosen_target: int | None = None
                rerouted = False
                preferred_blocker: int | None = None

                ok, blocker = _try_spill(out, bg, water, rr, cc, targets[preferred])
                if ok:
                    chosen_side = preferred
                    chosen_target = targets[preferred]
                else:
                    preferred_blocker = blocker
                    ok, _ = _try_spill(out, bg, water, rr, cc, targets[other])
                    if ok:
                        chosen_side = other
                        chosen_target = targets[other]
                        rerouted = True

                if collect_stats:
                    event = {
                        "hit_index": int(hits + 1),
                        "row": int(rr),
                        "col": int(cc),
                        "preferred": preferred,
                        "preferred_target": targets[preferred],
                        "other_target": targets[other],
                        "chosen": chosen_side,
                        "rerouted": bool(rerouted),
                        "preferred_blocker": preferred_blocker,
                    }
                    source_events.append(event)
                    stats["total_hits"] = int(stats["total_hits"]) + 1
                    if rerouted:
                        stats["rerouted_hits"] = int(stats["rerouted_hits"]) + 1

                if chosen_side is None or chosen_target is None:
                    break

                hits += 1
                cc = int(chosen_target)
                continue

            if int(g[nr, cc]) == int(bg) and int(out[nr, cc]) != int(bg):
                break

            rr = nr
            _paint_vertical(out, bg, water, rr, cc)

        if collect_stats:
            stats["sources"].append(
                {
                    "source_index": int(source_idx),
                    "start_col": int(start_c),
                    "hits": int(hits),
                    "events": source_events,
                }
            )

    return out, stats


def _run_original(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, water, shelf = _detect_roles(g)
    out = g.copy()
    sources = _get_sources(g, water)
    shelf_segs = _build_shelf_segments(g, shelf)

    def find_shelf_seg(r: int, c: int) -> tuple[int, int] | None:
        return _find_shelf_seg(shelf_segs[int(r)], int(c))

    seen: set[tuple[int, int]] = set()
    q = deque((int(r), int(c)) for r, c in sources)

    while q:
        r, c = q.popleft()
        if (int(r), int(c)) in seen:
            continue
        seen.add((int(r), int(c)))

        rr, cc = int(r), int(c)
        out[rr, cc] = int(water)

        while True:
            nr = int(rr + 1)
            if nr >= h:
                break
            below = int(g[nr, cc])

            if below == int(shelf):
                seg = find_shelf_seg(nr, cc)
                if seg is None:
                    break
                a, b = seg
                nominal_left = int(max(0, a - 1))
                nominal_right = int(min(w - 1, b + 1))
                if int(g[rr, cc]) == int(bg):
                    out[rr, cc] = int(water)
                eff_left = int(cc)
                for x in range(int(cc) - 1, nominal_left - 1, -1):
                    if int(g[rr, x]) != int(bg):
                        break
                    eff_left = int(x)
                    out[rr, x] = int(water)
                eff_right = int(cc)
                for x in range(int(cc) + 1, nominal_right + 1):
                    if int(g[rr, x]) != int(bg):
                        break
                    eff_right = int(x)
                    out[rr, x] = int(water)
                if eff_left == nominal_left and nominal_left != int(cc):
                    q.append((int(rr), int(nominal_left)))
                if eff_right == nominal_right and nominal_right != int(cc):
                    q.append((int(rr), int(nominal_right)))
                break

            rr = nr
            out[rr, cc] = int(water)

    return out


def analyze_behavior(grid: np.ndarray) -> dict[str, object]:
    combo_out, stats = _run_combo(grid, collect_stats=True)
    original_out = _run_original(grid)
    return {
        "source_order": list(stats["source_order"]),
        "source_hits": [int(src["hits"]) for src in stats["sources"]],
        "rerouted_hits": int(stats["rerouted_hits"]),
        "total_hits": int(stats["total_hits"]),
        "diff_vs_original": int(np.count_nonzero(combo_out != original_out)),
        "source_events": stats["sources"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    out, _ = _run_combo(grid, collect_stats=False)
    return out


def _place_segment(grid: np.ndarray, row: int, a: int, b: int) -> None:
    grid[int(row), int(a) : int(b) + 1] = int(SHELF)


def _sample_layout() -> tuple[int, int, dict[str, int]] | None:
    c1 = random.randint(4, 7)
    d1 = random.randint(2, min(4, c1))
    t1 = int(c1 - d1)
    d2 = random.randint(5, 7)
    t2 = int(t1 + d2)

    w_min = max(18, t2 + 12)
    if w_min > 28:
        return None
    w = random.randint(w_min, 28)

    c2_lo = int(t2 + 5)
    c2_hi = int(w - 7)
    if c2_lo > c2_hi:
        return None
    c2 = random.randint(c2_lo, c2_hi)

    d4 = random.randint(2, 4)
    t4 = int(c2 + d4)
    if t4 > w - 3:
        return None

    d5_hi = min(4, int(w - 1 - t4))
    if d5_hi < 2:
        return None
    d5 = random.randint(2, d5_hi)
    t5 = int(t4 + d5)

    r1 = random.randint(3, 5)
    r2 = int(r1 + random.randint(4, 6))
    r3 = int(r2 + random.randint(4, 6))
    h = int(r3 + random.randint(4, 7))
    if h > 30:
        return None

    return h, w, {
        "c1": int(c1),
        "c2": int(c2),
        "r1": int(r1),
        "r2": int(r2),
        "r3": int(r3),
        "t1": int(t1),
        "t2": int(t2),
        "t4": int(t4),
        "t5": int(t5),
    }


def _gen_attempt() -> np.ndarray | None:
    sample = _sample_layout()
    if sample is None:
        return None
    h, w, p = sample

    grid = np.full((h, w), BG, dtype=int)
    grid[0, p["c1"]] = WATER
    grid[1, p["c1"]] = WATER
    grid[0, p["c2"]] = WATER
    grid[1, p["c2"]] = WATER

    s1 = int(p["r1"] + 1)
    s2 = int(p["r2"] + 1)
    s3 = int(p["r3"] + 1)

    a1 = int(p["t1"] + 1)
    b1 = random.randint(p["c1"], min(w - 2, p["c1"] + 3))

    a2 = random.randint(max(0, p["t1"] - 2), p["t1"])
    b2 = int(p["t2"] - 1)

    a4 = int(p["t2"] + 1)
    b4 = int(p["t4"] - 1)

    a5 = random.randint(max(0, p["t4"] - 2), p["t4"])
    b5 = int(p["t5"] - 1)

    if not (a1 <= p["c1"] <= b1):
        return None
    if not (a2 <= p["t1"] <= b2):
        return None
    if not (a4 <= p["c2"] <= b4):
        return None
    if not (a5 <= p["t4"] <= b5):
        return None

    rows_to_segments: dict[int, list[tuple[int, int]]] = {
        s1: [(a1, b1)],
        s2: [(a2, b2), (a4, b4)],
        s3: [(a5, b5)],
    }
    for row, segs in rows_to_segments.items():
        segs_sorted = sorted(segs)
        for i in range(len(segs_sorted) - 1):
            if int(segs_sorted[i][1]) + 1 >= int(segs_sorted[i + 1][0]):
                return None
        for a, b in segs_sorted:
            _place_segment(grid, row, a, b)

    combo_out = main(grid)
    if np.array_equal(combo_out, grid):
        return None

    metrics = analyze_behavior(grid)
    if metrics["source_order"] != sorted(metrics["source_order"]):
        return None
    if metrics["rerouted_hits"] < 1:
        return None
    if metrics["source_hits"][0] < 2:
        return None
    if metrics["source_hits"][1] < 2:
        return None
    if metrics["diff_vs_original"] <= 0:
        return None

    for row, segs in rows_to_segments.items():
        for a, b in segs:
            if not np.all(combo_out[int(row), int(a) : int(b) + 1] == int(SHELF)):
                return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 300 attempts")
