# Combo variant of de809cff (index 2)
# combo_spec: D1/2+D12/1
# Name: 凹痕大小决定补丁颜色+凹痕形状编码补丁方向

from __future__ import annotations

import random
import sys
from collections import Counter as _Counter
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _cand = _parent / "solvers"
    if _cand.is_dir():
        sys.path.append(str(_cand))
        break

from arc_utils import Color


# Design analysis
# a) 具体化与行为追踪
#    - 原题先清理两块大前景区域中的单像素噪声，再把每个黑色 dent 当成独立中心：
#      总是盖一个“对侧色 3x3 方块”，最后把中心改成 teal。
#    - 融合后不能再把每个黑像素独立处理，必须先把相邻黑像素分成 4 连通 dent 组。
#      组的“大小”控制颜色，组的“形状”控制补丁几何：
#      * 单像素组 -> 对侧色 3x3 方块，中心 teal。
#      * 2 像素水平组 -> 在 dent 外露一侧生成宿主色水平 3 格条纹，两个 dent 像素改成 teal。
#      * 2 像素垂直组 -> 在 dent 外露一侧生成宿主色垂直 3 格条纹，两个 dent 像素改成 teal。
#    - 小例子：若 clean 区域里一对水平 dent 位于 `(r,c),(r,c+1)`，原题会对这两个像素
#      各自盖 opposite-color 3x3，形成厚实的异色团块；融合后会在 dent 外侧单独放一条
#      宿主色水平条，并把 `(r,c),(r,c+1)` 改成 teal。差异不是单一维度，而是
#      “单像素/双像素”与“水平/垂直”同时参与决定输出。
# b) 复杂度校准
#    - 只保留三类 dent：single、horizontal pair、vertical pair，类别数足够少，
#      人类可以从 2-4 组样例归纳；同时又能明显区别于原题的“所有 dent 都变 3x3 方块”。
#    - 每个样例强制同时出现 single、horizontal pair、vertical pair，避免 size 或
#      shape 某一维在某张图里完全缺席。
#    - pair 仍是少数，保留原题整体观感；但两种 pair 方向都会重复出现，不会沦为偶发噪点。
# c) 对原 main() 隐式假设的审查
#    - 原 `_dent_centers()` 把每个黑像素独立判定为 dent 中心，这在融合后失效：
#      相邻黑像素必须先合并成组，否则会错误地产生两块 3x3。
#    - 原 stamping 逻辑默认“每个 dent 都是 opposite-color 3x3”，该前提已失效；
#      现在颜色取决于组大小，几何取决于组形状。
#    - 原逻辑默认黑像素之间互不相关。融合后 solver 新增假设：可转换的多像素 dent 只会
#      是 size==2 的正交相邻对，generator 只构造这三类合法组并保证组间分离。
# d) 放置合法性
#    - generator 继续采用 output-first：先造 clean 区域，再挑 dent 组，再反推输入黑洞。
#    - 不同 dent 组的黑像素之间保持 Chebyshev 距离至少 3，避免连成更大组件，也避免
#      补丁互相堆叠得难以阅读。
#    - 为了让“宿主色条纹”真的可见，pair 只能放在区域边界附近，并要求它至少有一侧外露。
#      条纹不覆盖 dent 本体，而是贴在外露一侧，因此会在区域外侧形成独立的 3 格信号。
# e) 视觉可观察性
#    - single 仍然会产生显眼的 opposite-color 3x3，pair 则留下 teal 对和外侧独立条纹，
#      同一张图里能直接看出“单像素 vs 双像素”的颜色差异。
#    - horizontal 与 vertical pair 的外侧条纹方向不同，所以方向编码来自可见几何，
#      不是颜色编号或处理顺序。
#    - 如果把 pair 放在宿主区内部，宿主色条纹会被背景吞没而几乎不可见；因此生成器显式
#      禁止这类退化布局。
# f) 可视化整洁度
#    - 不引入新颜色；仍只有背景、两种前景色和 teal 标记。
#    - pair 的可见新增部分是一条独立 3 格宿主色条，信息明确但不拥挤；全局分离约束避免 teal
#      标记与补丁叠成一团。
# g) 颜色变化可推断性
#    - 颜色规则只有两档：single -> 对侧色，pair -> 宿主色。配合 single 与 pair 在同色
#      区域内重复出现，人类可以从“有无异色 3x3 光晕”归纳 size 维度。
# 9) main() 的新隐式假设与 generator 保证
#    - 假设 1：可转换多像素 dent 只会是 2 像素水平/垂直 pair。generator 保证。
#    - 假设 2：pair 的 host 在整个组内一致。generator 只在同一区域内选 pair。
#    - 假设 3：pair 的条纹会贴在 clean 区域中暴露的一侧；若两侧都暴露，则用左侧/上侧打破
#      平局。generator 只生成至少有一侧暴露的 pair，使这个规则既稳定又可观察。
#    - 假设 4：背景噪声和边界串色不会贴着 dent 组，避免干扰 host 判定。generator 在加噪时
#      显式避开 dent 邻域。


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _has_same_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = g.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                if bool(np.all(g[rr : rr + 2, cc : cc + 2] == int(color))):
                    return True
    return False


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    Two sweeps are enough for the observed corruption pattern: the first removes
    protruding wrong-color cells, the second lets their neighboring true cells
    keep the corrected local majority.
    """
    h, w = g.shape
    a, b = map(int, colors)
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = bg
                    continue

                sub = cur[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
                own = int(np.sum(sub == v))
                other = b if v == a else a
                other_count = int(np.sum(sub == other))
                bg_count = int(np.sum(sub == bg))

                if _has_same_2x2(cur, r, c, v):
                    nxt[r, c] = v
                elif other_count > own and other_count >= 5:
                    nxt[r, c] = other
                elif bg_count >= 5 and own <= 4:
                    nxt[r, c] = bg
                elif own <= 2:
                    nxt[r, c] = bg
                else:
                    nxt[r, c] = v
        cur = nxt

    return cur


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _dent_host(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    r: int,
    c: int,
) -> int | None:
    if int(original[r, c]) != int(bg):
        return None
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    host = max(colors, key=lambda color: counts[int(color)])
    other = colors[1] if int(host) == int(colors[0]) else colors[0]
    if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
        return int(host)
    return None


def _group_shape(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) == 1:
        return "single"
    if len(cells) != 2:
        return None
    (r0, c0), (r1, c1) = sorted(cells)
    if r0 == r1 and abs(c0 - c1) == 1:
        return "horizontal"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "vertical"
    return None


def _dent_groups(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
) -> list[tuple[list[tuple[int, int]], int, str]]:
    h, w = original.shape
    candidates: dict[tuple[int, int], int] = {}
    for r, c in zip(*np.where(original == int(bg))):
        rr = int(r)
        cc = int(c)
        host = _dent_host(original, clean, colors, bg, rr, cc)
        if host is not None:
            candidates[(rr, cc)] = host

    groups: list[tuple[list[tuple[int, int]], int, str]] = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(candidates):
        if start in seen:
            continue
        stack = [start]
        comp: list[tuple[int, int]] = []
        hosts: set[int] = set()
        while stack:
            cell = stack.pop()
            if cell in seen:
                continue
            seen.add(cell)
            comp.append(cell)
            hosts.add(int(candidates[cell]))
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in candidates and (nr, nc) not in seen:
                    stack.append((nr, nc))

        shape = _group_shape(comp)
        if shape is None or len(hosts) != 1:
            continue
        groups.append((sorted(comp), next(iter(hosts)), shape))
    return groups


def _pair_cap_side(
    clean: np.ndarray,
    cells: list[tuple[int, int]],
    host: int,
    shape: str,
) -> str | None:
    h, w = clean.shape
    if shape == "horizontal":
        r = int(cells[0][0])
        c0 = min(int(c) for _r, c in cells)
        c1 = max(int(c) for _r, c in cells)
        left_exposed = not (0 <= c0 - 1 < w and int(clean[r, c0 - 1]) == int(host))
        right_exposed = not (0 <= c1 + 1 < w and int(clean[r, c1 + 1]) == int(host))
        if left_exposed or not right_exposed:
            return "left"
        return "right"
    if shape == "vertical":
        c = int(cells[0][1])
        r0 = min(int(r) for r, _c in cells)
        r1 = max(int(r) for r, _c in cells)
        up_exposed = not (0 <= r0 - 1 < h and int(clean[r0 - 1, c]) == int(host))
        down_exposed = not (0 <= r1 + 1 < h and int(clean[r1 + 1, c]) == int(host))
        if up_exposed or not down_exposed:
            return "up"
        return "down"
    return None


def _stamp_group(
    out: np.ndarray,
    cells: list[tuple[int, int]],
    shape: str,
    color: int,
    side: str | None = None,
) -> None:
    h, w = out.shape
    if shape == "single":
        r, c = cells[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(color)
        return
    if shape == "horizontal":
        r = int(cells[0][0])
        c0 = min(int(c) for _r, c in cells)
        start = c0 - 3 if side != "right" else c0 + 2
        for cc in range(start, start + 3):
            if 0 <= cc < w:
                out[r, cc] = int(color)
        return
    if shape == "vertical":
        c = int(cells[0][1])
        r0 = min(int(r) for r, _c in cells)
        start = r0 - 3 if side != "down" else r0 + 2
        for rr in range(start, start + 3):
            if 0 <= rr < h:
                out[rr, c] = int(color)
        return
    raise ValueError(f"unknown dent shape: {shape}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents inside two colored regions into teal-marked patches whose color and geometry are both classified from the dent group.

    Input structure:
    - A black background contains two large foreground-color regions.
    - The regions include small black dents plus one-cell foreground noise in the background or along the color boundary.
    - Relevant dents are either isolated black pixels or 2-cell horizontal/vertical black pairs.

    Rule:
    - First clean the foreground locally, removing isolated specks and undoing one-cell boundary contamination.
    - Then find black dent components whose cleaned 3x3 neighborhoods are clearly dominated by one host color, and classify each component by size and orientation.
    - A single-cell dent becomes an opposite-color 3x3 patch, while a 2-cell horizontal or vertical dent creates a host-color stripe on the exposed side in that same orientation. Finally recolor every original dent cell teal (8).

    Key insight:
    - The solver must group black cells before recoloring. Dent size chooses between opposite-color and same-side recoloring, and dent shape chooses between a square, a horizontal stripe, and a vertical stripe, so the output encodes two independent properties of the same defect.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    groups = _dent_groups(g, clean, colors, bg)
    for cells, host, shape in groups:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        stamp_color = int(other) if shape == "single" else int(host)
        side = _pair_cap_side(clean, cells, host, shape)
        _stamp_group(out, cells, shape, stamp_color, side)

    for cells, _host, _shape in groups:
        for r, c in cells:
            out[r, c] = Color.TEAL

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
PAIR_COUNT_MIN, PAIR_COUNT_MAX = 2, 2
MIN_SINGLE_PER_COLOR = 2

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint two overlapping rectangles with guaranteed black-backed left/top borders."""
    grid = np.zeros((N, N), dtype=int)

    a_h = rng.randint(int(N * 0.62), int(N * 0.70))
    a_w = rng.randint(int(N * 0.56), int(N * 0.64))
    a_r0 = rng.randint(max(2, N // 4 - 1), max(2, N // 4 + 1))
    a_c0 = 2
    a_r1 = min(N - 2, a_r0 + a_h)
    a_c1 = min(N - 4, a_c0 + a_w)

    b_h = rng.randint(int(N * 0.62), int(N * 0.70))
    b_w = rng.randint(int(N * 0.56), int(N * 0.64))
    b_r0 = 2
    b_c0 = rng.randint(max(3, N // 3), max(3, N // 3 + 2))
    b_r1 = min(N - 3, b_r0 + b_h)
    b_c1 = min(N - 2, b_c0 + b_w)

    if a_r1 - a_r0 < 8 or a_c1 - a_c0 < 6 or b_r1 - b_r0 < 6 or b_c1 - b_c0 < 8:
        return None

    grid[a_r0:a_r1, a_c0:a_c1] = color_a
    grid[b_r0:b_r1, b_c0:b_c1] = color_b

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    overlap = max(0, min(a_r1, b_r1) - max(a_r0, b_r0)) * max(0, min(a_c1, b_c1) - max(a_c0, b_c0))
    if overlap < max(20, N):
        return None
    return grid


def _is_safe_dent_cell(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = color
                if int(np.sum(block == color)) == 4:
                    return True
    return False


def _all_far_enough(cells: list[tuple[int, int]], chosen: list[list[tuple[int, int]]]) -> bool:
    for group in chosen:
        for r, c in cells:
            for rr, cc in group:
                if max(abs(r - rr), abs(c - cc)) < DENT_PATCH_SEPARATION:
                    return False
    return True


def _near_any_dent(r: int, c: int, dent_cells: set[tuple[int, int]]) -> bool:
    return any(max(abs(r - dr), abs(c - dc)) <= 1 for dr, dc in dent_cells)


def _enumerate_single_candidates(clean: np.ndarray, host: int, other: int) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    candidates: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if _is_safe_dent_cell(clean, r, c, host, other):
                candidates.append([(r, c)])
    return candidates


def _enumerate_pair_specs(
    clean: np.ndarray,
    host: int,
    shape: str,
) -> list[tuple[list[tuple[int, int]], str]]:
    h, w = clean.shape
    candidates: list[tuple[list[tuple[int, int]], str]] = []

    if shape == "horizontal":
        for r in range(1, h - 1):
            for c0 in range(0, w - 3):
                block = clean[r - 1 : r + 2, c0 : c0 + 4]
                if not bool(np.all(block == int(host))):
                    continue
                cells = [(r, c0 + 1), (r, c0 + 2)]
                candidates.append((cells, "left"))
                candidates.append((cells, "right"))
        return candidates

    if shape == "vertical":
        for r0 in range(0, h - 3):
            for c in range(1, w - 1):
                block = clean[r0 : r0 + 4, c - 1 : c + 2]
                if not bool(np.all(block == int(host))):
                    continue
                cells = [(r0 + 1, c), (r0 + 2, c)]
                candidates.append((cells, "up"))
                candidates.append((cells, "down"))
        return candidates

    raise ValueError(f"unknown pair shape: {shape}")


def _pick_from_candidates(
    candidates: list[list[tuple[int, int]]],
    n: int,
    rng: random.Random,
    chosen_global: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]] | None:
    pool = [list(cells) for cells in candidates]
    rng.shuffle(pool)
    picked: list[list[tuple[int, int]]] = []
    for cells in pool:
        if _all_far_enough(cells, chosen_global + picked):
            picked.append(cells)
            if len(picked) == n:
                return picked
    return None if n > 0 else []


def _pick_pair_specs(
    candidates: list[tuple[list[tuple[int, int]], str]],
    n: int,
    rng: random.Random,
    chosen_global: list[list[tuple[int, int]]],
) -> list[tuple[list[tuple[int, int]], str]] | None:
    pool = [(list(cells), side) for cells, side in candidates]
    rng.shuffle(pool)
    picked: list[tuple[list[tuple[int, int]], str]] = []
    for cells, side in pool:
        if _all_far_enough(cells, chosen_global + [grp for grp, _side in picked]):
            picked.append((cells, side))
            if len(picked) == n:
                return picked
    return None if n > 0 else []


def _carve_pair_cap(clean: np.ndarray, cells: list[tuple[int, int]], side: str) -> None:
    if side == "left":
        r = int(cells[0][0])
        c0 = min(int(c) for _r, c in cells)
        clean[r, c0 - 1] = Color.BLACK
        return
    if side == "right":
        r = int(cells[0][0])
        c1 = max(int(c) for _r, c in cells)
        clean[r, c1 + 1] = Color.BLACK
        return
    if side == "up":
        c = int(cells[0][1])
        r0 = min(int(r) for r, _c in cells)
        clean[r0 - 1, c] = Color.BLACK
        return
    if side == "down":
        c = int(cells[0][1])
        r1 = max(int(r) for r, _c in cells)
        clean[r1 + 1, c] = Color.BLACK
        return
    raise ValueError(f"unknown cap side: {side}")


def _enumerate_stable_boundary_pairs(
    clean: np.ndarray,
    host: int,
    shape: str,
) -> list[tuple[list[tuple[int, int]], str]]:
    h, w = clean.shape
    out: list[tuple[list[tuple[int, int]], str]] = []
    if shape == "horizontal":
        for r in range(1, h - 1):
            for c0 in range(1, w - 4):
                if int(clean[r, c0 - 1]) != Color.BLACK:
                    continue
                if not bool(np.all(clean[r - 1 : r + 2, c0 : c0 + 4] == int(host))):
                    continue
                out.append(([(r, c0 + 1), (r, c0 + 2)], "left"))
        return out
    if shape == "vertical":
        for r0 in range(1, h - 4):
            for c in range(1, w - 1):
                if int(clean[r0 - 1, c]) != Color.BLACK:
                    continue
                if not bool(np.all(clean[r0 : r0 + 4, c - 1 : c + 2] == int(host))):
                    continue
                out.append(([(r0 + 1, c), (r0 + 2, c)], "up"))
        return out
    raise ValueError(f"unknown shape: {shape}")


def _stamp_dent_groups(
    clean: np.ndarray,
    dent_groups_a: list[tuple[list[tuple[int, int]], str]],
    dent_groups_b: list[tuple[list[tuple[int, int]], str]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    output = clean.copy()
    for dent_groups, host, other in (
        (dent_groups_a, color_a, color_b),
        (dent_groups_b, color_b, color_a),
    ):
        for cells, shape in dent_groups:
            stamp_color = int(other) if shape == "single" else int(host)
            side = _pair_cap_side(clean, cells, host, shape)
            _stamp_group(output, cells, shape, stamp_color, side)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells, _shape in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    dent_cells: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    """Place isolated fg-color specks in background area."""
    bg_candidates = list(zip(*np.where(clean == 0)))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        if _near_any_dent(r, c, dent_cells):
            continue
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check that changing (r,c) to `other` won't cause the solver to misclassify any neighbor."""
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[r - nr0, c - nc0] = other
            own_n = int(np.sum(nsub == host))
            other_n = int(np.sum(nsub == other))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    dent_cells: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    """Place cross-color contamination at region-region boundaries."""
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (color_a, color_b):
                continue
            other = color_b if v == color_a else color_a
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((r, c, v, other))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if _near_any_dent(r, c, dent_cells):
            continue
        if int(inp[r, c]) != int(host):
            continue
        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == host))
        other_count = int(np.sum(sub_now == other))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = other
        placed += 1


def _plan_group_targets(
    n_a: int,
    n_b: int,
    color_a: int,
    color_b: int,
    availability: dict[int, dict[str, int]],
    rng: random.Random,
) -> tuple[dict[int, dict[str, int]], dict[int, int]] | None:
    max_pairs = min(PAIR_COUNT_MAX, n_a + n_b - 2 * MIN_SINGLE_PER_COLOR)
    if max_pairs < PAIR_COUNT_MIN:
        return None
    if availability[color_a]["horizontal"] + availability[color_b]["horizontal"] == 0:
        return None
    if availability[color_a]["vertical"] + availability[color_b]["vertical"] == 0:
        return None
    shapes = ["horizontal", "vertical"]
    rng.shuffle(shapes)

    pair_targets = {
        color_a: {"horizontal": 0, "vertical": 0},
        color_b: {"horizontal": 0, "vertical": 0},
    }
    singles = {color_a: n_a, color_b: n_b}

    for shape in shapes:
        choices = [
            color
            for color in (color_a, color_b)
            if singles[color] > MIN_SINGLE_PER_COLOR
            and pair_targets[color][shape] < availability[color][shape]
        ]
        if not choices:
            return None
        top = max((availability[color][shape] - pair_targets[color][shape], singles[color]) for color in choices)
        choices = [
            color
            for color in choices
            if (availability[color][shape] - pair_targets[color][shape], singles[color]) == top
        ]
        chosen = rng.choice(choices)
        pair_targets[chosen][shape] += 1
        singles[chosen] -= 1

    if singles[color_a] < MIN_SINGLE_PER_COLOR or singles[color_b] < MIN_SINGLE_PER_COLOR:
        return None
    if pair_targets[color_a]["horizontal"] + pair_targets[color_a]["vertical"] == 0 and (
        pair_targets[color_b]["horizontal"] + pair_targets[color_b]["vertical"] == 0
    ):
        return None
    return pair_targets, singles


def _choose_pair_groups_for_color(
    stable_specs: dict[str, list[tuple[list[tuple[int, int]], str]]],
    pair_targets: dict[str, int],
    rng: random.Random,
    chosen_global: list[list[tuple[int, int]]],
) -> list[tuple[list[tuple[int, int]], str, str]] | None:
    groups: list[tuple[list[tuple[int, int]], str, str]] = []

    picked_h = _pick_pair_specs(
        stable_specs["horizontal"],
        pair_targets["horizontal"],
        rng,
        chosen_global,
    )
    if picked_h is None:
        return None
    groups.extend((cells, "horizontal", side) for cells, side in picked_h)

    picked_v = _pick_pair_specs(
        stable_specs["vertical"],
        pair_targets["vertical"],
        rng,
        chosen_global + [cells for cells, _shape, _side in groups],
    )
    if picked_v is None:
        return None
    groups.extend((cells, "vertical", side) for cells, side in picked_v)
    return groups


def _choose_single_groups_for_color(
    clean: np.ndarray,
    host: int,
    other: int,
    single_count: int,
    rng: random.Random,
    chosen_global: list[list[tuple[int, int]]],
) -> list[tuple[list[tuple[int, int]], str]] | None:
    picked_single = _pick_from_candidates(
        _enumerate_single_candidates(clean, host, other),
        single_count,
        rng,
        chosen_global,
    )
    if picked_single is None:
        return None
    return [(cells, "single") for cells in picked_single]


def _group_counts(groups: list[tuple[list[tuple[int, int]], str]]) -> dict[str, int]:
    counts = {"single": 0, "horizontal": 0, "vertical": 0}
    for _cells, shape in groups:
        counts[shape] += 1
    return counts


def _group_recognized_in_clean(
    clean: np.ndarray,
    colors: tuple[int, int],
    cells: list[tuple[int, int]],
    host: int,
    shape: str,
) -> bool:
    trial = clean.copy()
    for r, c in cells:
        trial[r, c] = Color.BLACK
    groups = _dent_groups(trial, clean, colors, Color.BLACK)
    return any(group_cells == cells and int(group_host) == int(host) and group_shape == shape for group_cells, group_host, group_shape in groups)


def _all_groups_present(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    expected: list[tuple[list[tuple[int, int]], int, str]],
) -> bool:
    actual = _dent_groups(original, clean, colors, Color.BLACK)
    actual_set = {(tuple(cells), int(host), shape) for cells, host, shape in actual}
    expected_set = {(tuple(cells), int(host), shape) for cells, host, shape in expected}
    return expected_set.issubset(actual_set)


def _stable_pair_specs(
    clean: np.ndarray,
    colors: tuple[int, int],
    host: int,
    shape: str,
) -> list[tuple[list[tuple[int, int]], str]]:
    stable: list[tuple[list[tuple[int, int]], str]] = []
    for cells, side in _enumerate_pair_specs(clean, host, shape):
        trial = clean.copy()
        _carve_pair_cap(trial, cells, side)
        canonical = _clean_foreground(trial, colors, Color.BLACK)
        if _group_recognized_in_clean(canonical, colors, cells, host, shape):
            stable.append((cells, side))
    return stable


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None
    h_row = N // 2
    v_col = N // 2 + 3
    clean[h_row, 2] = Color.BLACK
    clean[2, v_col] = Color.BLACK

    dents_a = [[(h_row, 3), (h_row, 4)], [(N - 5, N // 2 - 1)]]
    dents_b = [[(3, v_col), (4, v_col)], [(N // 2 - 2, N - 5)]]

    inp = clean.copy()
    for cells in dents_a + dents_b:
        for r, c in cells:
            inp[r, c] = Color.BLACK

    groups = _dent_groups(inp, _clean_foreground(inp, (color_a, color_b), Color.BLACK), (color_a, color_b), Color.BLACK)
    expected = {
        (tuple([(h_row, 3), (h_row, 4)]), int(color_a), "horizontal"),
        (tuple([(3, v_col), (4, v_col)]), int(color_b), "vertical"),
        (tuple([(N - 5, N // 2 - 1)]), int(color_a), "single"),
        (tuple([(N // 2 - 2, N - 5)]), int(color_b), "single"),
    }
    actual = {(tuple(cells), int(host), shape) for cells, host, shape in groups}
    if expected != actual:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
