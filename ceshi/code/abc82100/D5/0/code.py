# Variant of abc82100
# Dimension: D5 — 模板放置的对齐方式 (transform)
# Change: d.anchor（规则示例中的对齐参考点）对齐到源像素s的位置，模板整体平移
#   → 模板中心对齐到源像素：规则示例仍用锚点来指向对应 teal 模板，但真正盖章时忽略锚点，
#   改为把模板包围盒的几何中心对齐到散落源像素。
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     old: legend 中 source->target 后方的 anchor 对齐到散落 source 像素
#     new: teal 模板的几何中心（这里具体化为模板包围盒中心）对齐到散落 source 像素
#   concrete_behavior:
#     - 输入仍由若干局部 legend 组成：一对 4 邻接的 source/target 彩色像素，沿同方向再前进一步
#       得到 anchor；anchor 贴着或落在一个 teal 连通模板上。
#     - 散落的 source 像素仍表示“在这里触发该规则”。
#     - 输出时，不再把 legend 的 anchor 挪到散落 source 上，而是把 teal 模板包围盒中心挪到
#       散落 source 上，再整体重着色为 target。
#   behavior_trace:
#     - 例：某模板 bbox 中心在局部坐标 (1,1)，anchor 在 (0,2)。
#     - 原题：散落 source 位于 (10,10) 时，模板会按 anchor 对齐，所以中心落在 (11,9)。
#     - 变体：同一 source 位于 (10,10) 时，模板中心直接落在 (10,10)。
#     - 差异：输出图案整体平移了 (anchor - center)；对非对称模板，人眼能直接看到位置变化。
#   complexity_calibration:
#     - 若模板是单像素、完全中心对称、或 anchor 恰好等于中心，则新旧规则输出相同，会退化。
#     - 因此生成器强制：
#       1) 模板至少 4 像素；
#       2) 模板不是关于 bbox 中心的完全对称图形；
#       3) anchor != bbox 中心；
#       4) 每条规则至少有一个散落 source 使完整“中心对齐”模板落在网格内，保证差异可见。
#   implicit_assumptions_audit:
#     - `_discover_rules` 仍依赖旧属性：anchor 在 target 的延长线上，且附近存在唯一 teal 组件。
#       该前提在变体中仍成立，因为 legend 只负责“指向哪一个模板”，不负责最终对齐方式。
#     - `main` 不能再用 `dr = sr - anchor_r` 的平移公式；改为用模板中心计算相对偏移。
#     - `main` 继续假设同一 source_color 只对应一条规则；生成器保持 source/target 配对唯一。
#     - `main` 继续假设散落 source 不参与 legend；生成器仍通过 occupied/reserved 分离它们。
#   placement_legality:
#     - legend 放置时仍保证 source、target、anchor 和 teal 模板都在边界内，且不同 legend 不接触。
#     - 散落 source 仍彼此隔离，避免伪造新的相邻 source->target 规则。
#     - 输出允许裁剪和重叠，但生成器至少保证每条规则有一个完整中心盖章，以便观察变体。
#   visual_observability:
#     - 变体只改空间对齐，不引入新的颜色排序或隐藏数值条件。
#     - 非对称模板 + 非中心 anchor 让“同色 source 生成的 target 图案位置”相对源像素明显变化。
#   cleanliness:
#     - 仍保留少量旁观像素，且颜色不与任何 source 色冲突。
#     - 保持 legend 与散落 source 分区稀疏，减少输入混杂。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_center(pattern: np.ndarray) -> tuple[int, int]:
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    return (int((int(rows.min()) + int(rows.max())) // 2), int((int(cols.min()) + int(cols.max())) // 2))


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "center": _pattern_center(np.asarray(pattern, dtype=int)),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn color-change examples from local source-target-teal groups and stamp their motifs elsewhere.

    Input structure:
    - Black background with isolated colored pixels and teal (8) connected template components.
    - Each legend example is a 4-neighbor source->target pair whose forward continuation reaches
      an anchor touching the teal component for that rule.

    Rule:
    - For every discovered legend, recolor its teal component to the target color and copy it onto
      every non-legend pixel of the same source color.
    - The copied component is positioned by aligning the teal component's geometric center
      (its bounding-box center) to the scattered source pixel, not by aligning the legend anchor.
    - Erase all teal pixels and all colored pixels that served only as legend examples.

    Key insight: The adjacent colored pair tells you which source color becomes which target color,
    while the nearby teal component supplies the shape; the source pixel marks the center of that shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    # Preserve ordinary colored pixels that are not consumed by any learned rule.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        cr, cc = rule["center"]
        pattern = rule["pattern"]
        rel_rows = pattern[:, 0] - int(cr)
        rel_cols = pattern[:, 1] - int(cc)
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            rows = rel_rows + sr
            cols = rel_cols + sc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 6, 2: 10, 3: 15, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size (scaled by grid area)
STENCIL_MAX_CAP = 12
STENCIL_MIN_PIXELS = 4
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
STENCIL_RETRIES = 400

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 80
FULL_STAMP_RETRIES = 100
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

# Cardinal directions (4-connected)
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 8-connected neighbor offsets
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _normalize_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in coords)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    todo = [next(iter(coords))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for dr, dc in _DIRS8:
            nxt = (int(r) + dr, int(c) + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return len(seen) == len(coords)


def _bbox_center_local(coords: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return ((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2)


def _is_centrally_symmetric(coords: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    max_r = max(rows)
    max_c = max(cols)
    coords_set = set(coords)
    return all((max_r - r, max_c - c) in coords_set for r, c in coords)


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    for _ in range(STENCIL_RETRIES):
        coords = {(0, 0)}
        while len(coords) < n:
            r, c = random.choice(list(coords))
            dr, dc = random.choice(_DIRS8)
            coords.add((int(r) + dr, int(c) + dc))

        coords_list = _normalize_coords(coords)
        rows = [r for r, _ in coords_list]
        cols = [c for _, c in coords_list]
        height = max(rows) + 1
        width = max(cols) + 1
        if height % 2 == 0 or width % 2 == 0:
            continue
        if height == 1 and width == 1:
            continue
        if not _is_connected(set(coords_list)):
            continue
        if _is_centrally_symmetric(coords_list):
            continue

        center = _bbox_center_local(coords_list)
        anchor_candidates = [p for p in coords_list if p != center]
        if not anchor_candidates:
            continue
        anchor = max(anchor_candidates, key=lambda p: (_chebyshev(p, center), abs(p[0] - center[0]) + abs(p[1] - center[1])))
        return coords_list, anchor, center

    raise RuntimeError("Failed to build non-symmetric centered stencil")


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(STENCIL_MIN_PIXELS, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_MIN_PIXELS, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor, center = _make_stencil(min_s, max_s)
        rel_anchor = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rel_center = [(int(r - center[0]), int(c - center[1])) for r, c in stencil]
        center_span = {
            "min_r": min(r for r, _ in rel_center),
            "max_r": max(r for r, _ in rel_center),
            "min_c": min(c for _, c in rel_center),
            "max_c": max(c for _, c in rel_center),
        }
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel_anchor": rel_anchor,
                "rel_center": rel_center,
                "center_span": center_span,
                "anchor": anchor,
                "center": center,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _place_legends(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel_anchor"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((int(nr), int(nc)))
            if not ok:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _can_fit_full_centered(rule, r: int, c: int, H: int, W: int) -> bool:
    span = rule["center_span"]
    return (
        0 <= int(r) + int(span["min_r"])
        and int(r) + int(span["max_r"]) < H
        and 0 <= int(c) + int(span["min_c"])
        and int(c) + int(span["max_c"]) < W
    )


def _place_scattered(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    H, W = inp.shape
    for rule in rules:
        positions: list[tuple[int, int]] = []

        full_stamp_pos = None
        for _ in range(FULL_STAMP_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            if not _can_fit_full_centered(rule, r, c, H, W):
                continue
            full_stamp_pos = (int(r), int(c))
            break
        if full_stamp_pos is None:
            return False

        positions.append(full_stamp_pos)
        occupied.add(full_stamp_pos)
        _reserve_4neighbors(reserved, [full_stamp_pos])

        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])

        if len(positions) < SCATTERED_MIN:
            return False

        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) not in occupied and (r, c) not in reserved:
                bc = random.choice(bystander_colors)
                inp[r, c] = bc
                placed.append((int(r), int(c), int(bc)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(int(r), int(c))])
                break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int, *, use_center: bool = True) -> np.ndarray:
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        rel = rule["rel_center"] if use_center else rule["rel_anchor"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                nr = int(sr) + int(rr)
                nc = int(sc) + int(cc)
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = int(rule["tgt"])
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, H, W, use_center=True)
    old_output = _build_output(rules, bystander_pos, H, W, use_center=False)

    # Reject degenerate cases where the sampled instance still behaves like the original anchor-aligned task.
    if np.array_equal(output, old_output):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after %d attempts" % OUTER_RETRIES)
