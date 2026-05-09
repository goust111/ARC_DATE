# Combo variant of 36a08778 (index 29)
# combo_spec: D1/0+D8/0+D10/0
# Name: 对角线下落+架子溶解+水源排序
#
# 设计分析
# a) 具体化与行为追踪
#    取一个最小交互例子：左水源是两格斜线 [(0,2),(1,3)]，右水源是 [(0,10),(1,9)]。
#    两条 45 度水流分别沿右下 / 左下前进，并在第 5 行的同一根横架子第 6 列相遇。
#    原题里两条流都是竖直下落，碰到架子会在架子上一层水平扩散，且各水源并行处理，
#    所以不会出现“先开洞，后穿洞”的先后依赖。
#    融合后：左源先处理，撞到第 5 行架子后，不是竖直停住，而是沿自己的前进方向在架子上滑行，
#    同时把经过的架子像素溶解为背景；右源后处理时再走到第 5 行同一列，发现那个撞击点已被先前
#    水源溶掉，于是不再沿架子滑行，而是直接穿过缺口继续对角线下落，去撞更低的一层架子。
#    关键差异不只在“路径是斜的”，而在“斜线路径 + 可变架子 + 顺序处理”三者同时决定后续轨迹。
#
# b) 复杂度校准
#    规则只引入两个离散方向 dc in {-1,+1}，避免太多自由度；架子仍保持“水平连续段”，视觉对象类型
#    没变。为了避免退化成“只是把竖直改成斜线”，生成器固定构造共享首撞架子和后续次级架子，使每个样例
#    都至少出现一次“后源穿过前源溶出的孔洞”。这样既有明显新行为，又不会把对象种类扩到难以归纳。
#
# c) 隐式假设审查
#    - 原 main() 假设 source 是竖直 1x2 柱，故从 bottom row 直接开始、dc=0。融合后 source 必须能编码
#      对角线方向，所以改为读取顶部连通水块的顶行/底行列偏移，推断 dc=±1。
#    - 原 main() 每一步都从原始 g 读 shelf，默认架子静态不变。D8/0 打破这个前提，因此必须维护可变 world，
#      每次命中都在当前 world 上重新找连续架子段。
#    - 原 main() 用一个全局 BFS/visited 近似并行处理，默认水源互不影响。D10/0 打破此点，因此改为按列排序
#      顺序处理，并把已溶解孔洞保留给后续水源。
#    - 原 main() 的 visited 只看 (r,c) 就够，因为全是竖直状态。现在同一点可能来自不同对角方向，状态至少
#      要包含 dc。
#    - 原逻辑默认“架子命中后只在架子上一层画水，不改架子像素”。现在命中过程本身会删改架子，而且删除的是
#      架子上的一段前缀/后缀，所以 segment lookup 不能预先缓存。
#    - D10/0 原始描述还提到前序轨迹可阻挡后序轨迹。这里保留一个较弱但可控的版本：只有先前画在原背景上的
#      水迹会阻挡后续水源；被溶解出的架子孔洞不会被当作阻挡，这样“后源穿洞”才真正可发生。
#
# d) 放置合法性
#    生成器显式构造：
#    - 两个顶部 8 连通斜水源，不重叠且与边界保留 1 格余量；
#    - 一条共享首撞架子，保证左源先削掉右半段、右源随后正好踏过被削掉的撞击点；
#    - 左右各自至少还有一条更低的次级架子，保证后续轨迹可见；
#    - 所有架子都只放在空背景上，且不覆盖源像素、不越界。
#
# e) 视觉可观察性
#    人类看到 2-4 个样例时，能直接观察到：
#    - 水源不是竖条而是斜条，因此下落方向可见；
#    - 某些红架子在输出中被水“咬掉一截”；
#    - 靠左的源总是先动，因为靠右的源会利用前者留下的缺口继续下探到更低层，而不是在共享架子处横滑。
#    这些都是纯空间结构线索，不依赖颜色编号以外的隐藏信息。
#
# f) 可视化整洁度
#    只保留三类颜色：背景、品红水、红架子。没有额外标记层。共享交互只发生在一条主架子上，左右后续路径
#    各自分开，避免多条水流在同一小区域反复覆盖导致难以辨认。
#
# g) 颜色变化可推断性
#    本融合不引入新的颜色映射，仍是“水源颜色 = 输出水迹颜色，架子颜色 = 可被擦除对象颜色”。因此颜色规则
#    与原题一致，变化只来自几何与处理顺序，人类容易归纳。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import components_mask


BG = 7
WATER = 6
SHELF = 2


class Source(NamedTuple):
    row: int
    col: int
    dc: int
    cells: frozenset[tuple[int, int]]


class SourceStats:
    def __init__(self, start: tuple[int, int], dc: int):
        self.start = (int(start[0]), int(start[1]))
        self.dc = int(dc)
        self.path_cells: list[tuple[int, int]] = []
        self.dissolved_cells: list[tuple[int, int]] = []
        self.hit_count = 0
        self.passed_prior_holes = 0
        self.stopped_by_trail = False


def _detect_roles(grid: np.ndarray) -> tuple[int, int, int]:
    cnt = Counter(map(int, grid.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non_bg = [int(c) for c in cnt if int(c) != int(bg)]
    assert len(non_bg) >= 2
    water = int(min(non_bg, key=lambda c: (int(cnt[int(c)]), int(c))))
    others = [int(c) for c in non_bg if int(c) != int(water)]
    shelf = int(max(others, key=lambda c: (int(cnt[int(c)]), -int(c))))
    return bg, water, shelf


def _extract_sources(grid: np.ndarray, water: int) -> list[Source]:
    W = grid.shape[1]
    sources: list[Source] = []
    for coords in components_mask(grid == int(water), connectivity=8):
        if coords.size == 0 or not bool((coords[:, 0] == 0).any()):
            continue
        rows = coords[:, 0]
        rmax = int(rows.max())
        rmin = int(rows.min())
        bottom_cols = sorted(int(c) for r, c in coords if int(r) == int(rmax))
        top_cols = sorted(int(c) for r, c in coords if int(r) == int(rmin))
        if not bottom_cols:
            continue
        bottom_col = int(bottom_cols[len(bottom_cols) // 2])
        top_col = int(round(sum(top_cols) / len(top_cols))) if top_cols else int(bottom_col)
        dc = int(np.sign(bottom_col - top_col))
        if dc == 0:
            dc = 1 if bottom_col < W // 2 else -1
        cells = frozenset((int(r), int(c)) for r, c in coords)
        sources.append(Source(row=rmax, col=bottom_col, dc=dc, cells=cells))
    return sources


def _segment_at(world: np.ndarray, row: int, col: int, shelf: int) -> tuple[int, int] | None:
    if not (0 <= int(row) < world.shape[0] and 0 <= int(col) < world.shape[1]):
        return None
    if int(world[int(row), int(col)]) != int(shelf):
        return None
    a = int(col)
    b = int(col)
    while a - 1 >= 0 and int(world[int(row), int(a - 1)]) == int(shelf):
        a -= 1
    while b + 1 < world.shape[1] and int(world[int(row), int(b + 1)]) == int(shelf):
        b += 1
    return int(a), int(b)


def _trace_source(
    world: np.ndarray,
    out: np.ndarray,
    bg: int,
    water: int,
    shelf: int,
    source: Source,
    trail_blockers: set[tuple[int, int]],
    prior_holes: set[tuple[int, int]],
    record_stats: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], SourceStats | None]:
    H, W = world.shape
    rr, cc = int(source.row), int(source.col)
    local_blockers: set[tuple[int, int]] = set()
    local_holes: set[tuple[int, int]] = set()
    seen: set[tuple[int, int, int]] = set()
    stats = SourceStats(start=(rr, cc), dc=int(source.dc)) if record_stats else None

    out[rr, cc] = int(water)

    while True:
        state = (int(rr), int(cc), int(source.dc))
        if state in seen:
            break
        seen.add(state)

        nr = int(rr + 1)
        nc = int(cc + source.dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        cell = int(world[nr, nc])
        if cell == int(shelf):
            seg = _segment_at(world, nr, nc, shelf)
            if seg is None:
                break
            a, b = seg
            if stats is not None:
                stats.hit_count += 1

            if int(source.dc) > 0:
                shelf_cols = list(range(int(nc), int(b) + 1))
                overflow = int(b + 1) if int(b) + 1 < W else None
            else:
                shelf_cols = list(range(int(nc), int(a) - 1, -1))
                overflow = int(a - 1) if int(a) - 1 >= 0 else None

            last_col = int(cc)
            for sc in shelf_cols:
                world[nr, sc] = int(bg)
                out[nr, sc] = int(water)
                local_holes.add((int(nr), int(sc)))
                last_col = int(sc)
                if stats is not None:
                    stats.path_cells.append((int(nr), int(sc)))
                    stats.dissolved_cells.append((int(nr), int(sc)))

            if overflow is not None:
                if (int(nr), int(overflow)) in trail_blockers:
                    if stats is not None:
                        stats.stopped_by_trail = True
                    break
                out[nr, overflow] = int(water)
                local_blockers.add((int(nr), int(overflow)))
                if stats is not None:
                    stats.path_cells.append((int(nr), int(overflow)))
                rr, cc = int(nr), int(overflow)
            else:
                rr, cc = int(nr), int(last_col)
            continue

        if cell != int(bg):
            break

        if (int(nr), int(nc)) in trail_blockers and (int(nr), int(nc)) not in prior_holes:
            if stats is not None:
                stats.stopped_by_trail = True
            break

        out[nr, nc] = int(water)
        if (int(nr), int(nc)) not in prior_holes:
            local_blockers.add((int(nr), int(nc)))
        elif stats is not None:
            stats.passed_prior_holes += 1
        if stats is not None:
            stats.path_cells.append((int(nr), int(nc)))
        rr, cc = int(nr), int(nc)

    return local_blockers, local_holes, stats


def _simulate(
    grid: np.ndarray,
    *,
    carry_mutation: bool,
    record_stats: bool,
) -> tuple[np.ndarray, list[SourceStats]]:
    g = np.array(grid, dtype=int)
    bg, water, shelf = _detect_roles(g)
    sources = sorted(_extract_sources(g, water), key=lambda src: (int(src.col), int(src.row)))

    out = g.copy()
    persistent_world = g.copy()
    trail_blockers: set[tuple[int, int]] = set()
    prior_holes: set[tuple[int, int]] = set()
    all_stats: list[SourceStats] = []

    for source in sources:
        world = persistent_world if carry_mutation else g.copy()
        source_prior_holes = prior_holes if carry_mutation else set()
        local_blockers, local_holes, stats = _trace_source(
            world=world,
            out=out,
            bg=bg,
            water=water,
            shelf=shelf,
            source=source,
            trail_blockers=trail_blockers,
            prior_holes=source_prior_holes,
            record_stats=record_stats,
        )
        trail_blockers |= local_blockers
        if carry_mutation:
            prior_holes |= local_holes
        if stats is not None:
            all_stats.append(stats)

    return out, all_stats


def main(grid: np.ndarray) -> np.ndarray:
    out, _ = _simulate(grid, carry_mutation=True, record_stats=False)
    return out


def _place_segment(
    grid: np.ndarray,
    row: int,
    a: int,
    b: int,
    occupied: set[tuple[int, int]],
) -> bool:
    if not (0 <= int(row) < grid.shape[0] and 0 <= int(a) <= int(b) < grid.shape[1]):
        return False
    cells = [(int(row), int(c)) for c in range(int(a), int(b) + 1)]
    if any(cell in occupied for cell in cells):
        return False
    for r, c in cells:
        grid[r, c] = int(SHELF)
        occupied.add((int(r), int(c)))
    return True


def _safe_distractor(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for _ in range(60):
        row = random.randint(2, H - 2)
        length = random.randint(2, min(5, W - 2))
        a = random.randint(1, W - length - 1)
        b = a + length - 1
        cells = {(int(row), int(c)) for c in range(int(a), int(b) + 1)}
        if cells & occupied or cells & forbidden:
            continue
        for r, c in cells:
            grid[r, c] = int(SHELF)
            occupied.add((int(r), int(c)))
        return True
    return False


def _build_candidate() -> np.ndarray | None:
    H = random.randint(12, 20)
    W = random.randint(14, 22)
    grid = np.full((H, W), int(BG), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for _ in range(120):
        r1_hi = min(7, H - 6)
        if r1_hi < 4:
            return None
        r1 = random.randint(4, r1_hi)
        left_reach = int(r1 - 1)

        left_tail = random.randint(2, 4)
        right_tail = random.randint(2, 4)

        c1_min = 2
        c1_max = W - 2 * left_reach - 4
        if c1_min > c1_max:
            continue
        c1 = random.randint(c1_min, c1_max)
        c2 = int(c1 + 2 * left_reach)
        if c2 > W - 3:
            continue

        meet_col = int(c1 + left_reach)
        a1 = int(meet_col - left_tail)
        b1 = int(meet_col + right_tail)
        if a1 < 1 or b1 > W - 2:
            continue

        exit1 = int(b1 + 1)

        max_d2 = min(4, H - r1 - 3, meet_col - 2)
        if max_d2 < 2:
            continue
        d2 = random.randint(2, max_d2)
        r2 = int(r1 + d2)
        impact2 = int(meet_col - d2)
        max_left2 = min(3, impact2 - 1)
        max_right2 = min(2, W - impact2 - 2)
        if max_left2 < 1 or max_right2 < 1:
            continue
        left2 = random.randint(1, max_left2)
        right2 = random.randint(1, max_right2)
        a2 = int(impact2 - left2)
        b2 = int(impact2 + right2)

        left_shelf_ok = False
        r3 = a3 = b3 = -1
        max_d3 = min(4, H - r1 - 3, W - exit1 - 2)
        if max_d3 >= 2:
            d3 = random.randint(2, max_d3)
            r3 = int(r1 + d3)
            impact3 = int(exit1 + d3)
            max_left3 = min(2, impact3 - 1)
            max_right3 = min(3, W - impact3 - 2)
            if max_left3 >= 1 and max_right3 >= 2:
                left3 = random.randint(1, max_left3)
                right3 = random.randint(2, max_right3)
                a3 = int(impact3 - left3)
                b3 = int(impact3 + right3)
                left_shelf_ok = True
        if not left_shelf_ok:
            continue

        src_cells = {
            (0, int(c1 - 1)),
            (1, int(c1)),
            (0, int(c2 + 1)),
            (1, int(c2)),
        }
        if any(not (0 <= c < W) for _, c in src_cells):
            continue

        grid.fill(int(BG))
        occupied.clear()
        for r, c in src_cells:
            grid[int(r), int(c)] = int(WATER)
            occupied.add((int(r), int(c)))

        if not _place_segment(grid, r1, a1, b1, occupied):
            continue
        if not _place_segment(grid, r2, a2, b2, occupied):
            continue
        if not _place_segment(grid, r3, a3, b3, occupied):
            continue

        forbidden: set[tuple[int, int]] = set()
        for step in range(1, H):
            row = int(1 + step)
            if row >= H:
                break
            lc = int(c1 + step)
            rc = int(c2 - step)
            if 0 <= lc < W:
                forbidden.add((row, lc))
            if 0 <= rc < W:
                forbidden.add((row, rc))
        for c in range(a1, b1 + 2):
            if 0 <= c < W:
                forbidden.add((int(r1), int(c)))
        for c in range(a2 - 1, b2 + 2):
            if 0 <= c < W:
                forbidden.add((int(r2), int(c)))
        for c in range(a3 - 1, b3 + 2):
            if 0 <= c < W:
                forbidden.add((int(r3), int(c)))

        if random.random() < 0.7:
            _safe_distractor(grid, occupied, forbidden)

        out, stats = _simulate(grid, carry_mutation=True, record_stats=True)
        control, _ = _simulate(grid, carry_mutation=False, record_stats=False)
        if len(stats) < 2:
            continue

        hole_reuse = int(sum(s.passed_prior_holes for s in stats[1:]))
        dissolved = int(sum(len(s.dissolved_cells) for s in stats))
        diff_pixels = int(np.sum(out != control))

        if np.array_equal(out, grid):
            continue
        if hole_reuse < 1:
            continue
        if diff_pixels < 4:
            continue
        if dissolved < 5:
            continue
        if stats[0].hit_count < 1 or stats[1].hit_count < 1:
            continue
        if int(np.sum(grid == int(SHELF))) <= int(np.sum(out == int(SHELF))):
            continue
        if int(np.sum(grid == int(SHELF))) <= int(np.sum(grid == int(WATER))):
            continue
        return grid.copy()

    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("Failed to generate valid input for combo D1/0+D8/0+D10/0 after 500 attempts")
