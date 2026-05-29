# Combo variant of de809cff (index 32)
# combo_spec: D7/1+D18/1
# Name: 重叠保留原色+凹痕密集成簇

from __future__ import annotations

import random
import sys
from collections import Counter as _Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import label

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import Color


"""
design_analysis:
  a_concretization_and_trace:
    - 原题链路是：清理两块大前景区域中的单像素噪声 -> 找到黑色 dent 中心 -> 每个 dent 都把自己的
      3x3 邻域整块刷成对侧前景色 -> 最后 dent 中心单独刷成 teal。
    - 本融合版保留清噪和 dent 判定，但把“刷补丁”改成两阶段：先统计所有 dent 的 3x3 覆盖次数，
      再只给覆盖次数恰好为 1 的像素刷对侧色；任何被两个及以上补丁共同覆盖的像素，都保留 clean
      网格中的原色不变，最后 dent 中心仍统一标成 teal。
    - 具体小例子：若同一前景区域里有两个横向相邻的黑点 `(r,c)` 与 `(r,c+1)`，原题会把并起来的
      3x4 区域几乎全部刷成对侧色，只在两个中心留 teal；融合后，这两个 3x3 补丁中间重叠的 2 列
      不再改色，仍保持宿主前景色，只有左右外缘的非重叠列被刷成对侧色，因此输出会出现“外圈翻色、
      中条保留、中心 teal”的簇状形态。
  b_complexity_calibration:
    - 致密簇只允许 2-4 个 dent，并使用少量连通模板（多米诺、三连、L、2x2、T），让规律有重复支撑，
      但不会引入过多互不相干的特殊形状。
    - 不允许不同簇彼此发生 3x3 级别重叠；这样所有“保留原色”的区域都能被肉眼归因为局部 dent 簇，
      不会在整张图上扩散成杂乱冲突。
    - generator 拒绝与旧版“直接覆盖”规则输出相同的样本，确保融合行为不是名义上的。
  c_implicit_assumption_audit:
    - `_main_colors()` 仍假设只有两种主要前景色；generator 仍只构造两种非黑非 teal 前景色。
    - `_clean_foreground()` 依赖真实区域足够厚、噪声主要是单像素；因此区域继续使用大矩形，噪声也仍限制
      为背景孤立色斑和边界单像素串色。
    - `_dent_centers()` 原先默认 dent 之间大多分散；融合后这个前提失效，所以不再用“顺次写补丁”的
      隐式覆盖规则，而是显式计算所有 dent 补丁的覆盖计数。
    - dent 判定仍要求每个黑点的 clean 3x3 中宿主色占多数且至少 5 格；generator 只在深内层放置
      2-4 格小簇，并逐簇验证同簇黑化后每个格子仍满足该判定。
  d_placement_legality:
    - 每个 dent 簇都必须完全位于单一前景区域内部，且每个成员的原始 3x3 邻域全是宿主色，避免越界、
      误宿主或把重叠效果和边界溢出混在一起。
    - 不同簇之间强制保持至少一个“补丁不相交”间隔，即任意跨簇中心对都不能同时满足
      `|dr|<=2 and |dc|<=2`。
    - 所有样本都做 oracle 校验：`main(inp)` 必须等于按融合规则生成的目标输出。
  e_visual_observability:
    - 输入里能直接看到 2-4 个成团出现的黑色 dent；输出里这些簇会变成多个 teal 中心，加上一圈
      对侧色外沿，但簇中央大片区域保持原宿主色，和原题“整块翻色”有明显区别。
    - 多个样本中都会重复出现“相邻 dent 越多，簇中央保留得越多”的现象，人类可从 2-4 组样例归纳出
      真正规则是“重叠补丁失效”而非“覆盖顺序不同”。
  f_visual_cleanliness:
    - 重叠只发生在局部簇内部，簇与簇分离；区域背景和两种前景大块结构仍然整洁。
    - teal 只承担“原 dent 中心”的语义，重叠保留只影响周围环带，不额外引入新标记颜色。
  g_inferability:
    - 颜色规则仍只有两层：宿主 dent 的 3x3 倾向翻成另一种前景色，中心标 teal。新增逻辑不依赖颜色
      数值，而依赖人眼可见的几何事实“补丁是否彼此重叠”。
    - generator 额外统计重叠保留单元数与新旧规则差异单元数；若这些指标为 0，就判定样本退化并重采样。
"""


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


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
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


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = clean[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)]
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _patch_coverage(shape: tuple[int, int], centers: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    coverage = np.zeros((h, w), dtype=int)
    for r, c, _host in centers:
        coverage[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] += 1
    return coverage


def _render_output(
    clean: np.ndarray,
    centers: list[tuple[int, int, int]],
    colors: tuple[int, int],
    *,
    preserve_overlap: bool,
) -> np.ndarray:
    h, w = clean.shape
    out = clean.copy()
    coverage = _patch_coverage(clean.shape, centers)
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        if preserve_overlap:
            mask = coverage[r0:r1, c0:c1] == 1
            out_patch = out[r0:r1, c0:c1]
            out_patch[mask] = int(other)
        else:
            out[r0:r1, c0:c1] = int(other)

    for r, c, _host in centers:
        out[r, c] = Color.TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn clustered black dents into opposite-color patches whose overlaps cancel out.

    Input structure:
    - Black background with two large foreground-color regions built from overlapping rectangles.
    - The regions contain single-pixel black dents, but many dents now appear in tight 2-4 cell
      clusters instead of being evenly scattered.
    - A few isolated one-cell foreground specks and one-cell boundary contaminations are still present
      and must be ignored before reading the dents.

    Rule:
    - First clean the foreground locally: remove isolated specks back to black, and revert one-cell
      cross-boundary contamination to the surrounding region color.
    - Detect every original black dent whose cleaned 3x3 neighborhood is mostly one foreground color.
    - Each dent proposes a 3x3 patch in the opposite foreground color, but pixels claimed by two or
      more dent patches are not recolored at all and stay at their cleaned original color.
    - After resolving overlap this way, recolor every dent center itself to teal (8).

    Key insight: The transformation is no longer "stamp every 3x3 patch"; it is "stamp only the
    non-overlapping fringe of each 3x3 patch". Dense dent clusters therefore leave their shared center
    area unchanged while still marking each dent center in teal.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _render_output(clean, centers, colors, preserve_overlap=True)


# --------------- generator (combo) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.48, 0.78
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 110
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.86

# --- CFG: clustered dent placement ---
CLUSTERS_PER_COLOR_MIN, CLUSTERS_PER_COLOR_MAX = 2, 3
TOTAL_DENTS_MIN, TOTAL_DENTS_MAX = 12, 22
MIN_LARGE_CLUSTERS = 2
INTER_CLUSTER_PATCH_SEPARATION = 3  # distinct clusters must not have overlapping 3x3 patches

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 14
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.82
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80
MIN_OVERLAP_PRESERVE_CELLS = 6
MIN_NEW_OLD_DIFF_CELLS = 6


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice([2, 3])
    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    chosen_corners = corners[:n_rects]

    if n_rects == 2:
        rect_colors = [color_a, color_b]
        rng.shuffle(rect_colors)
    else:
        first = rng.choice([color_a, color_b])
        mid = color_b if first == color_a else color_a
        rect_colors = [first, mid, first]

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        m = rng.randint(0, RECT_MARGIN_MAX)
        r0 = m if corner[0] == "T" else max(0, N - h - m)
        c0 = m if corner[1] == "L" else max(0, N - w - m)
        return r0, c0, min(N, r0 + h), min(N, c0 + w)

    grid = np.zeros((N, N), dtype=int)
    for corner, color in zip(chosen_corners, rect_colors):
        r0, c0, r1, c1 = _make_rect(corner)
        grid[r0:r1, c0:c1] = color

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None

    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None

    if n_rects == 3:
        _, n_components = label(grid == rect_colors[0])
        if n_components < 2:
            return None
    return grid


def _is_deep_host(clean: np.ndarray, r: int, c: int, host: int) -> bool:
    h, w = clean.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    return bool(np.all(clean[r - 1 : r + 2, c - 1 : c + 2] == int(host)))


def _cluster_safe(clean: np.ndarray, cells: tuple[tuple[int, int], ...], host: int, other: int) -> bool:
    temp = clean.copy()
    for r, c in cells:
        temp[r, c] = Color.BLACK
    for r, c in cells:
        sub = _sub3x3(temp, r, c)
        host_count = int(np.sum(sub == int(host)))
        other_count = int(np.sum(sub == int(other)))
        if host_count < 5 or host_count <= other_count:
            return False
    return True


def _normalize_shape(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in offsets)
    min_c = min(c for _r, c in offsets)
    return tuple(sorted((r - min_r, c - min_c) for r, c in offsets))


def _all_shape_variants(base: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = set()
    cur = list(base)
    for _ in range(4):
        pts = tuple(cur)
        variants.add(_normalize_shape(pts))
        variants.add(_normalize_shape(tuple((r, -c) for r, c in pts)))
        cur = [(-c, r) for r, c in cur]
    return sorted(variants)


_BASE_CLUSTER_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
]
_CLUSTER_SHAPES: dict[int, list[tuple[tuple[int, int], ...]]] = {2: [], 3: [], 4: []}
for _base in _BASE_CLUSTER_SHAPES:
    for _variant in _all_shape_variants(_base):
        _CLUSTER_SHAPES[len(_variant)].append(_variant)
for _size in _CLUSTER_SHAPES:
    _CLUSTER_SHAPES[_size] = sorted(set(_CLUSTER_SHAPES[_size]))


def _enumerate_cluster_candidates(
    clean: np.ndarray, host: int, other: int
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    h, w = clean.shape
    out: dict[int, list[tuple[tuple[int, int], ...]]] = {2: [], 3: [], 4: []}
    seen: set[tuple[tuple[int, int], ...]] = set()
    for size, shapes in _CLUSTER_SHAPES.items():
        for shape in shapes:
            max_r = max(r for r, _c in shape)
            max_c = max(c for _r, c in shape)
            for r0 in range(h - max_r):
                for c0 in range(w - max_c):
                    cells = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in shape))
                    if cells in seen:
                        continue
                    if any(clean[r, c] != int(host) or not _is_deep_host(clean, r, c, host) for r, c in cells):
                        continue
                    if not _cluster_safe(clean, cells, host, other):
                        continue
                    seen.add(cells)
                    out[size].append(cells)
    return out


def _clusters_conflict(
    cells: tuple[tuple[int, int], ...], occupied: list[tuple[int, int]]
) -> bool:
    for r, c in cells:
        for rr, cc in occupied:
            if abs(r - rr) < INTER_CLUSTER_PATCH_SEPARATION and abs(c - cc) < INTER_CLUSTER_PATCH_SEPARATION:
                return True
    return False


def _sample_cluster_sizes(rng: random.Random) -> tuple[list[int], list[int]] | None:
    n_a = rng.randint(CLUSTERS_PER_COLOR_MIN, CLUSTERS_PER_COLOR_MAX)
    n_b = rng.randint(CLUSTERS_PER_COLOR_MIN, CLUSTERS_PER_COLOR_MAX)
    sizes_a = [rng.randint(2, 4) for _ in range(n_a)]
    sizes_b = [rng.randint(2, 4) for _ in range(n_b)]
    all_sizes = sizes_a + sizes_b
    total = sum(all_sizes)
    if not (TOTAL_DENTS_MIN <= total <= TOTAL_DENTS_MAX):
        return None
    if sum(size >= 3 for size in all_sizes) < MIN_LARGE_CLUSTERS:
        return None
    return sizes_a, sizes_b


def _choose_clusters_for_sizes(
    candidates: dict[int, list[tuple[tuple[int, int], ...]]],
    sizes: list[int],
    occupied: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[tuple[int, int], ...]] | None:
    sizes = sorted(sizes, reverse=True)
    for _ in range(PLACEMENT_ATTEMPTS):
        taken = list(occupied)
        picked: list[tuple[tuple[int, int], ...]] = []
        ok = True
        for size in sizes:
            pool = list(candidates[size])
            rng.shuffle(pool)
            for cells in pool:
                if _clusters_conflict(cells, taken):
                    continue
                picked.append(cells)
                taken.extend(cells)
                break
            else:
                ok = False
                break
        if ok:
            return picked
    return None


def _clusters_to_centers(
    clusters_a: list[tuple[tuple[int, int], ...]],
    clusters_b: list[tuple[tuple[int, int], ...]],
    color_a: int,
    color_b: int,
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for cells in clusters_a:
        centers.extend((r, c, int(color_a)) for r, c in cells)
    for cells in clusters_b:
        centers.extend((r, c, int(color_b)) for r, c in cells)
    return centers


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = speck_color
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or inp[nr, nc] != host:
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


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random) -> None:
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
        if inp[r, c] != host:
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


def _sample_metrics(
    inp: np.ndarray, clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> dict[str, int]:
    coverage = _patch_coverage(inp.shape, centers)
    center_mask = np.zeros(inp.shape, dtype=bool)
    for r, c, _host in centers:
        center_mask[r, c] = True
    overlap_preserve_cells = int(np.sum((coverage >= 2) & (~center_mask)))
    new_out = _render_output(clean, centers, colors, preserve_overlap=True)
    legacy_out = _render_output(clean, centers, colors, preserve_overlap=False)
    diff_cells = int(np.sum(new_out != legacy_out))
    return {
        "overlap_preserve_cells": overlap_preserve_cells,
        "new_vs_legacy_diff": diff_cells,
        "dent_count": len(centers),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    size_spec = _sample_cluster_sizes(rng)
    if size_spec is None:
        return None
    sizes_a, sizes_b = size_spec

    candidates_a = _enumerate_cluster_candidates(clean, color_a, color_b)
    candidates_b = _enumerate_cluster_candidates(clean, color_b, color_a)
    if any(len(candidates_a[size]) == 0 for size in sizes_a):
        return None
    if any(len(candidates_b[size]) == 0 for size in sizes_b):
        return None

    clusters_a = _choose_clusters_for_sizes(candidates_a, sizes_a, [], rng)
    if clusters_a is None:
        return None
    occupied = [cell for cells in clusters_a for cell in cells]
    clusters_b = _choose_clusters_for_sizes(candidates_b, sizes_b, occupied, rng)
    if clusters_b is None:
        return None

    centers = _clusters_to_centers(clusters_a, clusters_b, color_a, color_b)
    colors = (color_a, color_b)
    output = _render_output(clean, centers, colors, preserve_overlap=True)

    inp = clean.copy()
    for r, c, _host in centers:
        inp[r, c] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None

    metrics = _sample_metrics(inp, clean, centers, colors)
    if metrics["overlap_preserve_cells"] < MIN_OVERLAP_PRESERVE_CELLS:
        return None
    if metrics["new_vs_legacy_diff"] < MIN_NEW_OLD_DIFF_CELLS:
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
