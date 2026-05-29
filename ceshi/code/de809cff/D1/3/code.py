# Variant of de809cff
# Dimension: D1 — 核心变换规则：凹痕到3x3补丁 (transform)
# Change: 每个黑色凹痕的3x3邻域多数前景色为A时，凹痕变为3x3的前景色B补丁，中心为teal(8)
# ... → 颜色交换：凹痕附近的同侧色像素变为对侧色，对侧色像素变为同侧色

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the D1 variant
# ----------------------------------
# variant_design:
#   concretization:
#     input_scene:
#       - 仍然是黑底 + 两个大前景区域 + 若干黑色 dent + 少量单像素噪声。
#       - solver 先做的清噪逻辑保持原样；清噪后，dent 中心仍然是黑色。
#     transform_change:
#       - 原题: 每个 dent 把整个 3x3 直接刷成“另一种前景色”，最后中心改成 teal。
#       - 变体: 每个 dent 只翻转该 3x3 中“已经存在的前景像素”：
#         host 色 A -> B，other 色 B -> A，黑色保持黑色；最后中心改成 teal。
#       - 因为只有两种前景色，这个局部动作可视为“在 dent 的 3x3 上，对所有前景像素做 A<->B toggle”。
#     traced_example:
#       - 清噪后某个 3x3 为:
#           A A B
#           A 0 B
#           A A B
#       - 原题输出:
#           B B B
#           B 8 B
#           B B B
#       - 本变体输出:
#           B B A
#           B 8 A
#           B B A
#       - 差异在于：原来 patch 内已有的对侧色不会被覆盖成同色，而是一起被翻回 host 色。
#   complexity_calibration:
#     - 若 dent 全都埋在纯色区域内部，则新旧任务几乎看不出差别。
#     - 因此 generator 必须显式提高“3x3 内同时看见两种前景色”的 dent 比例，
#       让 swap 行为在少量样例里肉眼可见。
#     - 这个变体的主要复杂度来自 mixed dents 的双向换色，因此 generator 优先保证
#       “边界附近的单中心 dents”稳定出现，而不强求额外的 pair-dent 叠加。
#   implicit_assumption_review:
#     main_related_lines:
#       - _main_colors: 依赖输入里只有两种非黑前景色。generator 继续只采样两种颜色。
#       - _clean_foreground: 依赖噪声仍是单像素散点/边界突出。本变体未改噪声模型。
#       - _dent_centers: 依赖真正的 dents 在清噪后其 3x3 中某一前景色仍占多数。
#         generator 继续用 host_count>=5(不含中心) 的筛选保证这一点。
#       - 新的 _apply_dent_swaps: 依赖 dent 的作用是“对 patch 内现有前景做 toggle”，
#         而不是无条件覆盖。黑背景必须保持黑色，否则会退回原题式整块涂抹。
#       - 多个 dents 的 patch 可能相交，若逐个覆盖会产生顺序依赖；因此这里改为先统计
#         每个像素被多少个 dent 的 3x3 覆盖，再按奇偶次 toggle，一次性出图。
#   placement_legality:
#     - 仍要求 dent 候选点本身属于某个 host 区域，并且黑化后该点能被 solver 识别。
#     - 基础 dent centers 仍按原题的同组间距约束挑选；不同 dents 的 3x3 即使重叠，
#       也由 parity toggle 自然合成，不再需要额外的先后顺序约定。
#     - output-first oracle 保留：若某次构造导致 solver 结果不等于预期输出，则丢弃重试。
#   observability:
#     - 关键可见指标是“mixed dents”: dent 的 3x3 中除了 host 外还能看到另一前景色。
#     - generator 强制每种颜色尽量选取一批 mixed dents，并额外要求新输出和旧规则输出
#       至少有若干像素差异，防止样本退化成原题。
#   cleanliness:
#     - 区域/噪声布局、颜色数和 teal 标记都保留原题风格，因此人类仍能先学会“找 dents”，
#       再注意到输出不是整块反色 patch，而是局部双向换色。


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


def _apply_dent_swaps(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    """Toggle the two foreground colors inside every dent's 3x3 neighborhood."""
    h, w = clean.shape
    a, b = map(int, colors)
    cover = np.zeros((h, w), dtype=np.uint8)
    for r, c, _host in centers:
        cover[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] ^= 1

    out = clean.copy()
    odd = cover.astype(bool)
    mask_a = odd & (clean == a)
    mask_b = odd & (clean == b)
    out[mask_a] = b
    out[mask_b] = a

    for r, c, _host in centers:
        out[r, c] = Color.TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into local color-swap mosaics.

    Input structure:
    - Black background with two large foreground colors.
    - The large regions contain small black dents and a few one-cell color specks
      or protrusions near the region boundaries.

    Rule:
    - First clean the two foreground regions: isolated specks become black, and
      one-cell cross-boundary contamination is restored to its surrounding region.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly one
      foreground color toggles the two foreground colors inside that 3x3 window.
    - Only already-foreground cells swap colors; black cells stay black, and the
      dent center itself is finally recolored teal (8).

    Key insight: The black cells that survive cleanup are the true dents; each of
    them acts like a 3x3 local A<->B color toggle rather than a solid repaint.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _apply_dent_swaps(clean, centers, colors)


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
DENT_PAIR_PROB = 0.0
MIXED_DENT_OTHER_MIN = 1
MIXED_DENT_RATIO_MIN = 0.30

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: anti-degeneration ---
LEGACY_DIFF_PIXELS_MIN = 4

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    """Return the 3x3 neighborhood of (r,c), clipped to grid bounds."""
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles (simple occlusion, painter's algorithm)."""
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


def _dent_neighbor_counts(grid: np.ndarray, r: int, c: int, host: int, other: int) -> tuple[int, int]:
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    other_count = int(np.sum(sub == other))
    return host_count, other_count


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect it as a dent in host region."""
    if int(grid[r, c]) != int(host):
        return False
    host_count, other_count = _dent_neighbor_counts(grid, r, c, host, other)
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    """Check whether placing `color` at (r,c) would complete a 2x2 block of that color."""
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


def _far_from(r: int, c: int, chosen: list[tuple[int, int]], min_sep: int) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in chosen)


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Select n dent centers with Chebyshev spacing against both chosen and blocked cells."""
    blocked = [] if blocked is None else list(blocked)
    pool = candidates.copy()
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if _far_from(r, c, blocked, DENT_PATCH_SEPARATION) and _far_from(r, c, chosen, DENT_PATCH_SEPARATION):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _pick_visible_dents(
    all_candidates: list[tuple[int, int]],
    mixed_candidates: list[tuple[int, int]],
    n: int,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    min_mixed = min(len(mixed_candidates), max(1, int(np.ceil(n * MIXED_DENT_RATIO_MIN))))
    max_mixed = min(n, len(mixed_candidates))
    for want_mixed in range(max_mixed, min_mixed - 1, -1):
        chosen_mixed = _pick_dent_centers(mixed_candidates, want_mixed, rng, blocked)
        if chosen_mixed is None:
            continue
        remaining_pool = [p for p in all_candidates if p not in set(chosen_mixed)]
        chosen_rest = _pick_dent_centers(remaining_pool, n - want_mixed, rng, blocked + chosen_mixed)
        if chosen_rest is not None:
            return chosen_mixed + chosen_rest
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    """Optionally expand single-cell dents into 2-cell adjacent pairs (~13% chance each)."""
    h, w = clean.shape
    result: list[list[tuple[int, int]]] = []
    for r, c in dents:
        cells = [(r, c)]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(clean[nr, nc]) != int(host) or not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                blocked_others = [pt for pt in blocked if pt != (r, c)]
                if not _far_from(nr, nc, blocked_others, DENT_PATCH_SEPARATION):
                    continue
                same_color_others = [pt for pt in dents if pt != (r, c)]
                if not _far_from(nr, nc, same_color_others, DENT_PATCH_SEPARATION):
                    continue
                cells.append((nr, nc))
                break
        result.append(cells)
    return result


def _all_dent_centers(
    dent_groups_a: list[list[tuple[int, int]]], dent_groups_b: list[list[tuple[int, int]]], color_a: int, color_b: int
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for cells in dent_groups_a:
        for r, c in cells:
            centers.append((r, c, int(color_a)))
    for cells in dent_groups_b:
        for r, c in cells:
            centers.append((r, c, int(color_b)))
    return centers


def _dented_clean(
    clean: np.ndarray, dent_groups_a: list[list[tuple[int, int]]], dent_groups_b: list[list[tuple[int, int]]]
) -> np.ndarray:
    base = clean.copy()
    for cells in dent_groups_a + dent_groups_b:
        for r, c in cells:
            base[r, c] = Color.BLACK
    return base


def _stamp_dents(
    clean: np.ndarray, dent_groups_a: list[list[tuple[int, int]]], dent_groups_b: list[list[tuple[int, int]]], color_a: int, color_b: int
) -> np.ndarray:
    """Build output by toggling foreground colors inside dent 3x3s and marking teal centers."""
    centers = _all_dent_centers(dent_groups_a, dent_groups_b, color_a, color_b)
    return _apply_dent_swaps(clean, centers, (color_a, color_b))


def _legacy_stamp_dents(
    clean: np.ndarray, dent_groups_a: list[list[tuple[int, int]]], dent_groups_b: list[list[tuple[int, int]]], color_a: int, color_b: int
) -> np.ndarray:
    """Old task behavior, kept only to reject degenerate samples."""
    h, w = clean.shape
    out = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for cells in dent_groups_a + dent_groups_b:
        for r, c in cells:
            out[r, c] = Color.TEAL
    return out


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
    """Place isolated fg-color specks in background area."""
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


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random) -> None:
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
    mixed = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((r, c))
                if _dent_neighbor_counts(clean, r, c, color_a, color_b)[1] >= MIXED_DENT_OTHER_MIN:
                    mixed[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((r, c))
                if _dent_neighbor_counts(clean, r, c, color_b, color_a)[1] >= MIXED_DENT_OTHER_MIN:
                    mixed[color_b].append((r, c))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    dents_a = _pick_visible_dents(valid[color_a], mixed[color_a], n_a, [], rng)
    dents_b = _pick_visible_dents(valid[color_b], mixed[color_b], n_b, dents_a or [], rng)
    if dents_a is None or dents_b is None:
        return None

    base_centers = dents_a + dents_b
    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, base_centers, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, base_centers, rng)

    output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b)
    legacy_output = _legacy_stamp_dents(clean, groups_a, groups_b, color_a, color_b)
    if int(np.count_nonzero(output != legacy_output)) < LEGACY_DIFF_PIXELS_MIN:
        return None

    inp = _dented_clean(clean, groups_a, groups_b)

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
    raise RuntimeError("generate_input: failed after %d attempts" % MAX_GENERATION_ATTEMPTS)
