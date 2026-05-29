# Variant of de809cff
# Dimension: D18 — 多凹痕补丁重叠时的处理规则 (multi_step)
# Change: 当多个凹痕的3x3补丁区域重叠时，所有补丁同时生成，重叠区域的颜色由最后写入的像素决定... → 重叠区域使用teal(8)标记冲突

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for D18 (written before implementation so the generator and solver
# share one concrete contract instead of drifting apart).
#
# spec_variant:
#   preserved_core:
#     - 仍然是黑底 + 两个大面积前景色区域。
#     - 仍然先清理孤立色斑和单像素边界污染，再从原始黑像素里识别 dents。
#     - 每个 dent 仍然生成一个裁剪后的 3x3 对侧颜色补丁，dent 中心仍然是 teal(8)。
#   changed_core:
#     - 原题在补丁重叠时等价于顺序覆盖，最后写入者决定颜色。
#     - 现在改为“同时盖章”: 对任意像素统计有多少个 dent 补丁覆盖它。
#     - 覆盖数 == 0: 保持清理后的底图。
#     - 覆盖数 == 1: 采用该唯一补丁的对侧颜色。
#     - 覆盖数 >= 2: 该像素标为 teal(8)，不再属于任何一方颜色。
#     - 因而 teal 同时承担两种可见角色: dent 中心标记、补丁冲突标记。
#   concrete_trace:
#     - 例: 同一区域中两个 dent 位于 (r,c) 和 (r,c+2)，两个 3x3 补丁会共享中间一列 3 个像素。
#     - 原逻辑: 这一列仍只是普通对侧颜色，看不出补丁相撞。
#     - D18: 这 3 个共享像素全部改为 teal；两处中心 teal 来自 dent 本身，其余 teal 来自冲突。
#     - 解题者因此会观察到“比输入黑洞更多的 teal”，且这些额外 teal 正好落在多个 3x3 扩张的交汇处。
#   complexity_calibration:
#     - 不引入新的对象类别、排序规则或颜色映射，只改补丁合成阶段，避免偏离原题。
#     - 但若仍保持原题的中心间隔 >= 3，则几乎看不到重叠；因此生成器必须显式放置少量近邻 dent 对。
#     - 近邻 dent 对采用同色区域内轴向间距 2 的单元组：补丁会重叠，但 dent 本身不相邻，视觉上更接近原题。
#   hidden_assumption_audit:
#     - 原 main 中逐个赋值 `out[r0:r1, c0:c1] = other` 隐含“覆盖冲突无须单独记录”。
#       这个前提在 D18 不成立，必须先累计覆盖次数，再统一结算。
#     - 原 `_dent_centers` 用 clean 的 3x3 多数票判定 host，这一前提在 D18 仍成立，因为变化只发生在输出合成阶段。
#     - 原 generator 通过 `DENT_PATCH_SEPARATION = 3` 基本禁止不同 dents 的 3x3 相交；
#       这个前提会让 D18 退化回原题，所以必须改成“少量特许 dent 对允许距离 2，其余仍保持 >=3”。
#   legality_and_solver_alignment:
#     - 每个 dent cell 仍单独满足安全条件 `_is_safe_dent_center`，保证 solver 会识别它。
#     - 特许 dent 对之间、以及它们与普通 dent 之间，都维持 Chebyshev >= 3，避免把整片区域变成连续 teal 噪声。
#     - 特许 dent 对内部则故意保持轴向距离 2；检测用的是 clean 而不是最终输出，因此这类近邻 dents 仍可被稳定识别。
#   observability_and_tidiness:
#     - 每张图要求至少 1 组冲突 dent 对，并要求存在“非 dent 中心 teal”像素。
#     - 这样 2~4 个样例里都能看到 teal 的双重角色，且额外 teal 形成局部细条带，而不是随机噪点。


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


def _patch_coverages(
    shape: tuple[int, int], centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    cover = np.zeros((h, w), dtype=int)
    unique_color = np.full((h, w), -1, dtype=int)

    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)

        view = unique_color[r0:r1, c0:c1]
        unset = view == -1
        view[unset] = int(other)
        unique_color[r0:r1, c0:c1] = view
        cover[r0:r1, c0:c1] += 1

    return cover, unique_color


def _compose_output(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    out = clean.copy()
    cover, unique_color = _patch_coverages(clean.shape, centers, colors)

    lone = cover == 1
    conflict = cover >= 2
    out[lone] = unique_color[lone]
    out[conflict] = Color.TEAL

    for r, c, _host in centers:
        out[r, c] = Color.TEAL

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert black dents in the two regions into opposite-color patches with teal collisions.

    Input structure:
    - Black background with two large foreground regions in distinct colors.
    - The regions may contain black dents, isolated foreground specks in the background,
      and single-cell cross-color contamination near the region boundary.

    Rule:
    - First locally denoise the two foreground regions: isolated specks turn black and
      one-cell boundary contamination is reassigned to the surrounding region.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly one foreground
      color generates a clipped 3x3 patch in the opposite foreground color.
    - All dent patches are combined simultaneously: cells covered by exactly one patch
      take that patch color, while cells covered by two or more patches become teal (8).
      Dent centers are also teal (8).

    Key insight: teal does not mark only the original black holes anymore; it also marks
    every pixel where multiple opposite-color dent patches overlap after the cleanup step.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _compose_output(clean, centers, colors)


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
COLLISION_PAIR_DISTANCE = 2
COLLISION_PAIR_COUNT_MIN, COLLISION_PAIR_COUNT_MAX = 1, 3
MIN_EXTRA_CONFLICT_TEAL = 3

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 600


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


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
        from scipy.ndimage import label

        _, n_components = label(grid == rect_colors[0])
        if n_components < 2:
            return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if grid[r, c] != host:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == other))


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


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    count: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = candidates[:]
    rng.shuffle(pool)
    picked: list[tuple[int, int]] = []
    anchors = list(blocked)
    for r, c in pool:
        cell = (int(r), int(c))
        if cell in blocked:
            continue
        if all(_cheb(cell, other) >= DENT_PATCH_SEPARATION for other in anchors):
            picked.append(cell)
            blocked.add(cell)
            anchors.append(cell)
            if len(picked) == count:
                return picked
    return None


def _is_detectable_dent_set(
    clean: np.ndarray, dents: list[tuple[int, int]], host: int, other: int
) -> bool:
    temp = clean.copy()
    for r, c in dents:
        if temp[r, c] != host:
            return False
        temp[r, c] = 0
    for r, c in dents:
        sub = _sub3x3(temp, r, c)
        host_count = int(np.sum(sub == host))
        other_count = int(np.sum(sub == other))
        if host_count < DENT_MIN_HOST_3x3 or host_count <= other_count:
            return False
    return True


def _collision_pair_candidates(
    clean: np.ndarray, host: int, other: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = clean.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if clean[r, c] != host or not _is_safe_dent_center(clean, r, c, host, other):
                continue
            for dr, dc in ((0, COLLISION_PAIR_DISTANCE), (COLLISION_PAIR_DISTANCE, 0)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if clean[rr, cc] != host or not _is_safe_dent_center(clean, rr, cc, host, other):
                    continue
                pair = [(r, c), (rr, cc)]
                if _is_detectable_dent_set(clean, pair, host, other):
                    candidates.append(((r, c), (rr, cc)))
    return candidates


def _pick_collision_pairs(
    pair_candidates_by_color: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]],
    quota: dict[int, int],
    colors: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] | None:
    max_pairs = min(
        COLLISION_PAIR_COUNT_MAX,
        quota[int(colors[0])] // 2 + quota[int(colors[1])] // 2,
    )
    if max_pairs < COLLISION_PAIR_COUNT_MIN:
        return None

    pool = [(color, pair) for color, pairs in pair_candidates_by_color.items() for pair in pairs]
    targets = list(range(COLLISION_PAIR_COUNT_MIN, max_pairs + 1))
    rng.shuffle(targets)
    targets.sort(reverse=True)

    for target in targets:
        remaining = quota.copy()
        used: set[tuple[int, int]] = set()
        chosen: list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = []
        shuffled = pool[:]
        rng.shuffle(shuffled)
        for color, pair in shuffled:
            if len(chosen) >= target or remaining[color] < 2:
                continue
            members = [pair[0], pair[1]]
            if any(_cheb(cell, other) < DENT_PATCH_SEPARATION for cell in members for other in used):
                continue
            chosen.append((color, pair))
            remaining[color] -= 2
            used.update(members)
            if len(chosen) == target:
                return chosen
    return None


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


def _variant_metrics(
    clean: np.ndarray,
    centers: list[tuple[int, int, int]],
    colors: tuple[int, int],
) -> dict[str, int]:
    cover, _ = _patch_coverages(clean.shape, centers, colors)
    center_mask = np.zeros(clean.shape, dtype=bool)
    for r, c, _host in centers:
        center_mask[r, c] = True
    conflict = cover >= 2
    return {
        "dent_cells": len(centers),
        "conflict_pixels": int(np.sum(conflict)),
        "extra_conflict_teal": int(np.sum(conflict & ~center_mask)),
        "max_cover": int(np.max(cover)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    h, w = clean.shape
    valid = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((r, c))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    pair_candidates = {
        color_a: _collision_pair_candidates(clean, color_a, color_b),
        color_b: _collision_pair_candidates(clean, color_b, color_a),
    }
    if not pair_candidates[color_a] and not pair_candidates[color_b]:
        return None

    quota = {color_a: n_a, color_b: n_b}
    collision_pairs = _pick_collision_pairs(pair_candidates, quota, (color_a, color_b), rng)
    if collision_pairs is None:
        return None

    blocked: set[tuple[int, int]] = set()
    centers: list[tuple[int, int, int]] = []
    remaining = quota.copy()
    for host, pair in collision_pairs:
        centers.append((pair[0][0], pair[0][1], host))
        centers.append((pair[1][0], pair[1][1], host))
        blocked.update(pair)
        remaining[host] -= 2

    dents_a = _pick_dent_centers(valid[color_a], remaining[color_a], blocked, rng)
    if dents_a is None:
        return None
    centers.extend((r, c, color_a) for r, c in dents_a)

    dents_b = _pick_dent_centers(valid[color_b], remaining[color_b], blocked, rng)
    if dents_b is None:
        return None
    centers.extend((r, c, color_b) for r, c in dents_b)

    metrics = _variant_metrics(clean, centers, (color_a, color_b))
    if metrics["extra_conflict_teal"] < MIN_EXTRA_CONFLICT_TEAL or metrics["max_cover"] < 2:
        return None

    output = _compose_output(clean, centers, (color_a, color_b))

    inp = clean.copy()
    for r, c, _host in centers:
        inp[r, c] = 0

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
